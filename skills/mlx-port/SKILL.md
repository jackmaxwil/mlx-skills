---
name: mlx-port
description: Porting Hugging Face and PyTorch models to MLX in mlx-lm style — the ModelArgs dataclass and silently dropped config keys, idempotent sanitize, weight renames and expert stacking, conv and conv-transpose layouts, the attention, mask and RoPE templates, cache choice for full, sliding, linear and SSM layers, batched array offsets, MoE routing with stop_gradient, dtype discipline inside a model, quant_predicate and cast_predicate, shard and pipeline hooks, custom architectures through model_file. Load when adding or fixing a model architecture, or translating PyTorch modules to mlx.nn.
---

# Porting a model to MLX

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. Read the transformers `modeling_*.py` for the architecture first. Then copy the closest existing file in `mlx_lm/models/` and change it, rather than writing one from nothing: `llama.py` for dense models, `qwen3_moe.py` for a softmax-routed MoE, `deepseek_v3.py` for MLA plus group-limited routing, and `qwen3_next.py` for a hybrid of attention and gated delta layers. Verification (greedy token diff against transformers, the `test_models` runner) belongs to `mlx-verify`.

## The surface mlx-lm expects

`mlx_lm.load` imports `mlx_lm.models.<model_type>`, after `MODEL_REMAPPING` in `utils.py` renames aliases (for example `mistral` → `llama`). The module must define:

| Name | Contract |
| --- | --- |
| `ModelArgs(BaseModelArgs)` | `@dataclass`; built by `ModelArgs.from_dict(config)` |
| `Model(nn.Module)` | `__call__(inputs, cache=None, input_embeddings=None)` returns logits `(B, L, vocab)` |
| `Model.layers` | property; its length sets how many default caches are made |
| `Model.sanitize(weights)` | optional; dict in, dict out; runs on **every** load |
| `Model.make_cache()` | optional; required when any layer is not a plain full-attention `KVCache` |
| `Model.quant_predicate`, `Model.cast_predicate` | optional properties (see "Quantization hooks") |
| `Model.shard(group)`, `model.model.pipeline(group)` | optional; `sharded_load` detects them with `hasattr` |

Load order is: build the model from the config, then `sanitize`, then `nn.quantize` the layers that have `.scales` in the checkpoint, then `load_weights(strict=True)`, then `mx.eval` the parameters. Strict loading reports mismatches with these errors:

```
ValueError: Received 1 parameters not in model: ...
ValueError: Missing 3 parameters: ...
ValueError: Expected shape (64,) but received shape (32,) for parameter model.norm.weight
```

Do not "fix" these by passing `strict=False`. That loads whatever happens to match and leaves the rest at random init.

## ModelArgs drops unknown keys silently

`BaseModelArgs.from_dict` keeps only the keys that are dataclass fields. A misspelled field, or a value nested in a sub-dict, keeps its default value without any warning:

```python
@dataclass
class ModelArgs(BaseModelArgs):
    hidden_size: int
    rope_theta: float = 10000.0

ModelArgs.from_dict({"hidden_size": 8, "rope_parameters": {"rope_theta": 1e6}})
# ModelArgs(hidden_size=8, rope_theta=10000.0)  <- the real 1e6 was lost
```

- Give every field that changes the math no default, so a missing key raises a `TypeError` at construction. Keep defaults only for fields that really are optional (`attention_bias`, `mlp_bias`, `rope_scaling`).
- Flatten nested configs in `__post_init__`. Newer transformers configs put `rope_theta` and the scaling dict inside `rope_parameters`, and multimodal configs put the language model inside `text_config`.
- Derive the defaulted values there as well: `num_key_value_heads = num_key_value_heads or num_attention_heads`, `head_dim = head_dim or hidden_size // num_attention_heads`, and `layer_types` for interleaved sliding windows.

## sanitize

`sanitize` receives the raw safetensors dict. It must give the same result for a Hugging Face checkpoint and for an MLX checkpoint that `mlx_lm.convert` already wrote, because both go through it.

- **Detect the source format.** Do not transform blindly:

```python
# Wrong: a converted checkpoint has no experts.N keys, and this raises KeyError.
for e in range(n): w.pop(f"{p}.experts.{e}.up_proj.weight")

# Right: find the HF-format layers first, and return early when there are none.
moe_layers = sorted(int(k.split(".")[2]) for k in w
                    if k.startswith("model.layers.") and k.endswith(".mlp.experts.0.up_proj.weight"))
if not moe_layers:
    return w
```

  Any one-shot arithmetic (`v + 1.0` for a norm stored as an offset from 1, a transpose) must sit after that early return or behind a shape guard. Otherwise each conversion applies it again.

- **Stack the experts** into the `SwitchGLU` layout: `w[f"{p}.mlp.switch_mlp.{n}.weight"] = mx.stack([... for e in range(num_experts)])`. Prequantized Hugging Face checkpoints also carry `scales` and `biases` per expert. Stack those suffixes too, as `deepseek_v3.py` does, or the quantized load fails.
- **Drop** precomputed `rotary_emb.inv_freq`, multi-token-prediction layers (`mtp.*`, or layers numbered past `num_hidden_layers`), and vision or audio towers in text-only ports.
- **Tied heads.** When `tie_word_embeddings`, pop `lm_head.weight` and compute logits with `self.model.embed_tokens.as_linear(h)`.
- **fp8 checkpoints** with `weight_scale_inv` dequantize inside `sanitize` (block-wise multiply after `mx.from_fp8`). `load` keeps mxfp4, mxfp8, nvfp4, AWQ/GPTQ and compressed-tensors checkpoints packed and maps them to MLX modes itself.

### Conv layouts

MLX convolutions are channels-last. PyTorch weights must be permuted, and inputs are `(B, L, C)` or `(B, H, W, C)`:

| Layer | PyTorch weight | MLX weight | Convert |
| --- | --- | --- | --- |
| `Conv1d` | `(out, in/groups, k)` | `(out, k, in/groups)` | `w.swapaxes(1, 2)` |
| `Conv2d` | `(out, in, kh, kw)` | `(out, kh, kw, in)` | `w.transpose(0, 2, 3, 1)` |
| `ConvTranspose1d` | `(in, out, k)` | `(out, k, in)` | `w.transpose(1, 2, 0)` |
| `ConvTranspose2d` | `(in, out, kh, kw)` | `(out, kh, kw, in)` | `w.transpose(1, 2, 3, 0)` |

Guard the permutation by shape so it stays idempotent. For a depthwise conv1d, PyTorch's last axis is `k` and MLX's last axis is 1: `if "conv1d.weight" in k and v.shape[-1] != 1: v = v.moveaxis(2, 1)`.

## The attention template

```python
def __call__(self, x, mask=None, cache=None):
    B, L, _ = x.shape
    q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
    else:
        q, k = self.rope(q), self.rope(k)
    o = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
    return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
```

- Call `mlx_lm.models.base.scaled_dot_product_attention`, not `mx.fast` directly. The wrapper switches to the quantized-cache path when the cache has `.bits`. That path does not support `sinks`.
- **Do not repeat KV heads for GQA.** Pass `k` and `v` with `n_kv_heads` heads. Repeating them gives the same result but copies the cache every step.
- Apply q/k norms (`q_norm`, `k_norm`) **before** RoPE, after the transpose, when the reference does.
- The scale is `head_dim**-0.5` unless the config overrides it (`query_pre_attn_scalar`, MLA's mscale). A wrong scale gives plausible but wrong text.
- An additive mask must have the output dtype. A float32 additive mask with bf16 queries raises `[scaled_dot_product_attention] Mask type must promote to output type bfloat16.` Use boolean masks, or cast the mask to `q.dtype`.
- `mx.fast.scaled_dot_product_attention` in mlx 0.32.4 takes `scale`, `mask`, `sinks` and `force_fused`. There is no `window_size` argument in 0.32.4, so express sliding windows through the mask (below).

### Masks

Build masks once in the model body, before the layer loop, from the cache of a layer of the right kind:

```python
fa_mask = create_attention_mask(h, cache[self.fa_idx])
swa_mask = create_attention_mask(h, cache[self.swa_idx], window_size=self.sliding_window)
```

`create_attention_mask` returns `None` for a single-token step, the string `"causal"` for an unwindowed prefill, and an array when a window, left padding or a rotating cache needs one. Never build `mx.arange(offset, offset + L)` from `cache.offset` yourself, for the reason below.

### Offsets can be arrays

In batched generation `BatchKVCache.offset` is an `mx.array` with one value per sequence (`[2, 3]` for left padding `[1, 0]` after a 3-token step). `nn.RoPE` and `mx.fast.rope` accept an array offset. Python-side use of the offset does not: `mx.arange(cache.offset, ...)` raises a `TypeError`, and `int(cache.offset)` raises `ValueError: [convert] Only length-1 arrays can be converted to Python scalars.` Single-sequence tests pass either way, because a plain `KVCache.offset` is an `int`. Custom RoPE variants (YaRN, longrope, M-RoPE) must pass `offset` through to `mx.fast.rope` and must not branch on its value.

### RoPE

Use `mlx_lm.models.rope_utils.initialize_rope(head_dim, base, traditional, scaling_config, max_position_embeddings)`. It supports `default`, `linear`, `dynamic`, `llama3`, `yarn`/`deepseek_yarn`, `longrope`, `proportional` and `mrope`, and raises `Unsupported RoPE type` for anything else.

- `traditional=False` matches Hugging Face's `rotate_half` layout, which almost every HF checkpoint uses. `traditional=True` is the interleaved GPT-J and original Meta layout. The wrong choice raises nothing and produces garbage.
- Partial rotary embedding is `nn.RoPE(int(head_dim * partial_rotary_factor))`. The `dims` argument rotates only the leading `dims` features and passes the rest through unchanged.
- Precomputed frequencies go through `mx.fast.rope(x, dims, traditional=False, base=None, scale=1.0, offset=o, freqs=freqs)`. Store them under a `_freqs` attribute, because a leading underscore keeps them out of the parameter tree and out of `load_weights` (see `mlx`).

## Cache choice

| Layer kind | Cache | Notes |
| --- | --- | --- |
| Full attention | `KVCache()` | default when `make_cache` is absent |
| Sliding window | `RotatingKVCache(max_size=window)` | `to_quantized` raises `NotImplementedError`; not trimmable once `offset >= max_size` |
| Linear attention / SSM / gated delta | `ArraysCache(size=n)` | holds the conv state and recurrent state; call `cache.advance(S)` after the update |
| Several caches per layer | `CacheList(c1, c2)` | e.g. an indexer plus attention |
| Chunked local attention | `ChunkedKVCache(chunk)` | llama4 style |

```python
def make_cache(self):
    return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]
```

`load_prompt_cache` rebuilds caches by class name from `mlx_lm/models/cache.py`. A cache class defined in your model file can be saved but cannot be loaded back. Anything you store on a cache must be in `state` (evaluated after each prefill chunk) or reachable from the logits. A derived array that is reassigned each step and is not part of `state` grows an unevaluated graph without limit. That bug shipped in DeepSeek-V3.2's indexer and in `ArraysCache` metadata.

Recurrent kernels: `gated_delta_update(..., use_kernel=not self.training)`. The Metal kernel has no VJP, needs `Dk % 32 == 0`, and is skipped off-GPU. Kernel authoring belongs to `mlx-kernel`.

## MoE routing

```python
gates = mx.softmax(self.gate(x), axis=-1, precise=True)
inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
inds = mx.stop_gradient(inds)
scores = mx.take_along_axis(gates, inds, axis=-1)
if self.norm_topk_prob:
    scores = scores / scores.sum(axis=-1, keepdims=True)
y = (self.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2).astype(y.dtype)
```

- Without `stop_gradient`, any gradient through the block (LoRA, DWQ) fails with `ValueError: [gather_axis] Cannot calculate VJP with respect to indices. Use stop_gradient on indices to stop gradients from being computed.` Inference works fine without it, so the bug shows up only at training time. mlx-lm had to fix this in about thirty model files during 2026.
- Match the reference router exactly: softmax vs sigmoid scoring, whether a correction bias applies to selection only or to the weights too, and group-limited top-k (`deepseek_v3.py`).
- `SwitchGLU` sorts tokens by expert and calls `gather_mm` or `gather_qmm` with `sorted_indices`. Use it rather than a Python loop over experts.

## Dtype discipline inside a model

The model must return logits in the dtype of its weights. The `test_models` runner checks float32 and float16. Promotion rules live in `mlx`. The model-level traps are:

- `x * mx.array(2.0)` makes bf16 activations float32; `x * 2.0` keeps bf16. Store constants as Python floats, or create them with `dtype=x.dtype`.
- An fp32 parameter that `cast_predicate` leaves alone (a router bias) promotes everything it touches. Cast at the use site: `(scores + self.bias).astype(scores.dtype)`.
- Accumulate softmax and router scores in fp32 with `precise=True`, then cast back.

## Quantization hooks

```python
@property
def quant_predicate(self):
    def predicate(path, module):
        if path.endswith("mlp.gate"):
            return {"group_size": 64, "bits": 8}
        return True
    return predicate

@property
def cast_predicate(self):
    return lambda k: "e_score_correction_bias" not in k
```

`quant_predicate` is used by `mlx_lm.convert -q` and `quantize_model` unless the user passes one. `cast_predicate` keeps the listed parameters out of `convert --dtype`. Modes, bits and recipes belong to `mlx-quant`.

## Shard and pipeline hooks

Define `shard(group)`; `sharded_load` finds it with `hasattr`. For the sharding rules, see `mlx-distributed`.

For pipelining, mix `PipelineMixin` into the inner model, iterate `self.pipeline_layers`, and keep the layer indices so checkpoint keys still match. Backends, `sharded_load` and collective semantics belong to `mlx-distributed`.

## Custom architectures without editing mlx-lm

Put the module next to the weights and set `"model_file": "mymodel.py"` in `config.json`. It loads only with `trust_remote_code=True`. Without it, `load` refuses with `The model at ... requires importing and running a custom module ('mymodel.py') ...`, a gate added for CVE-2026-5843. The module must export `Model` and `ModelArgs`. For a pure alias of an existing architecture, a `MODEL_REMAPPING` entry is the whole port.

Upstream mlx-lm forbids agent-written PR descriptions, commits, pushes and review replies, and new models need the `new_model.md` PR template with reference output. The contribution policy itself belongs to `mlx`.
