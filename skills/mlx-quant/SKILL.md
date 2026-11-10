---
name: mlx-quant
description: MLX quantization — affine, mxfp4, nvfp4 and mxfp8 modes with their legal group sizes and bits and exact errors, the silent divisibility skip in mlx-lm, scale dtype and output dtype, nn.quantize predicates, routers, embeddings and tied heads, mixed recipes, dynamic_quant, DWQ, AWQ and GPTQ, nvfp4 global scales, activation quantization with QQLinear, saved dtypes and cast_predicate, dequantize, GGUF in and out, publishing to mlx-community. Load before quantizing a model, choosing a quantized checkpoint, or writing a quantization predicate.
---

# Quantization

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. KV-cache quantization belongs to `mlx-serve`, the qmv and qmm kernel routing to `mlx-hot-path`, and the `quant_predicate` hook in a model file to `mlx-port`.

## Modes

| Mode | `group_size` | `bits` | Stored | When |
| --- | --- | --- | --- | --- |
| `affine` (default) | 32, 64 (default), 128 | 1, 2, 3, 4 (default), 5, 6, 8 | packed `uint32` weight, float `scales` and `biases` | general weight-only quantization; the only mode the mixed recipes and learned methods use |
| `mxfp4` | 32 only | 4 only | `uint32` weight, `uint8` e8m0 scales, no biases | checkpoints published as mxfp4 (gpt-oss); interop |
| `nvfp4` | 16 only | 4 only | `uint8` scales, plus an optional float32 `global_scale` | NVIDIA ModelOpt NVFP4 checkpoints; activation quantization |
| `mxfp8` | 32 only | 8 only | `uint8` scales | fp8 checkpoints with e8m0 32×32 blocks; activation quantization |

`group_size=None` and `bits=None` take the mode's default, so `nn.quantize(model, mode="mxfp4")` gives group 32 and 4 bits. Passing an affine-style group size to an fp mode fails instead of being ignored:

```
[quantize] The requested group size 256 is not supported. The supported group sizes are 32, 64, and 128.
[quantize] The requested number of bits 7 is not supported. The supported bits are 1, 2, 3, 4, 5, 6 and 8.
[quantize] mxfp4 quantization requires group size 32 but got 64.
[quantize] nvfp4 quantization requires bits to be 4 but got 8.
[quantize] The last dimension of the matrix needs to be divisible by the quantization group size 64. However the provided  matrix has shape (8,100)
```

1-bit affine is new in mlx 0.32.4 and runs on both GPU and CPU. Smaller groups cost more bits per weight: with 16-bit scales and biases, affine overhead is `32 / group_size` bits, so 4-bit at group 64 is 4.5 bits per weight and at group 32 is 5.0.

## The divisibility trap

`nn.quantize` raises on the first layer whose input dimension does not divide by the group size. mlx-lm's `quantize_model`, which is what `mlx_lm.convert -q` calls, **skips that layer silently** and still writes the global `{"group_size": 64, "bits": 4}` into `config.json`. A model with a hidden size of 96 at group 64 keeps every q/k/v/o/gate/up projection in float, and the only sign is the printed line:

```
[INFO] Quantized model with 26.057 bits per weight.
```

Always read that number. 4-bit affine at group 64 should print about 4.5 to 5.0, depending on what stays in float. A double-digit value means layers were skipped, so rerun with `--q-group-size 32` (a dimension that is a multiple of 32 always divides). In code, check what is left in float:

```python
from mlx_lm.utils import quantize_model, compute_bits_per_weight

model, config = quantize_model(model, config, group_size=64, bits=4)
left = [p for p, m in model.named_modules() if type(m) in (nn.Linear, nn.Embedding)]
print(f"{compute_bits_per_weight(model):.2f} bpw; float layers: {left}")
```

Only the layers you deliberately kept in float should appear in that list. `SwitchLinear` experts count too, so add it to the `type(m)` check for MoE models.

## Scale dtype decides output dtype

Affine `quantized_matmul` returns `promote(x.dtype, scales.dtype)`. Quantizing float32 weights gives float32 scales, and then every bf16 activation that passes through comes out float32:

```python
wq, s, b = mx.quantize(w)                   # w is float32, so s and b are float32
mx.quantized_matmul(x_bf16, wq, s, b).dtype  # float32: the whole model now runs in fp32

model.set_dtype(mx.bfloat16)                 # cast first, so the scales are bf16
nn.quantize(model, group_size=64, bits=4)
```

`mlx_lm.convert` casts to `--dtype` (default: the config's `torch_dtype`) **before** quantizing, which is the right order. Hand-written code must do the same. fp modes return `x.dtype`.

## Predicates

`nn.quantize(model, group_size, bits, mode=, class_predicate=)` calls `class_predicate(path, module)` on every leaf module. It returns `False` to skip, `True` to use the call's settings, or a dict of `to_quantized` arguments for that layer. **By default it quantizes every module with `to_quantized`, and `nn.Embedding` has one.** With tied embeddings, quantizing `embed_tokens` also quantizes the output head, because logits come from `embed_tokens.as_linear`.

```python
def predicate(path, m):
    if not hasattr(m, "to_quantized"):
        return False
    if m.weight.shape[-1] % 64:
        return False                                   # or {"group_size": 32, ...}
    if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
        return {"group_size": 64, "bits": 8}           # MoE routers: mlx-lm's own predicates keep them at 8
    if path.endswith("lm_head") or path.endswith("embed_tokens"):
        return {"group_size": 64, "bits": 6}
    return True

nn.quantize(model, group_size=64, bits=4, class_predicate=predicate)
```

- A returned dict without `"mode"` means `affine`, even when the call passed `mode="mxfp4"`. Put the mode in the dict for every non-affine layer.
- `quantize_model` records every dict in `config["quantization"][path]`. The loader rebuilds exactly those shapes, so predicates do not need to run again at load time.
- Without an explicit predicate, mlx-lm uses the model's own `quant_predicate` property (Qwen3-MoE and Qwen3-Next keep the router at 8 bits). Passing your own predicate replaces it; it does not combine with it.
- Quantized layers freeze their weights. After `nn.quantize`, only biases of former `Linear` layers and non-quantized modules remain trainable (`mlx-train` for QLoRA).

## Choosing a recipe

| Goal | Command | Notes |
| --- | --- | --- |
| Quick 8-bit or 6-bit | `mlx_lm.convert --model M -q --q-bits 8` | upstream's DWQ notes treat 8-bit as usually as good as 16-bit |
| 4-bit, no calibration | `mlx_lm.convert --model M -q --quant-predicate mixed_4_6` | keeps `v_proj` and `down_proj` at 6 bits in the first and last eighth of layers and in every third layer, and `lm_head` at 6 bits; affine only |
| 3-bit or 2-bit | `mixed_3_6`, `mixed_3_4`, `mixed_2_6`, then DWQ | DWQ gives the largest gains at these widths |
| Hit a bits-per-weight budget | `mlx_lm.dynamic_quant --model M --target-bpw 4.5 --low-bits 4 --high-bits 5` | measures per-layer sensitivity on calibration data; fastest of the learned methods |
| Best 2–4-bit quality | `mlx_lm.dwq --model M --bits 4 --group-size 32` | distills from the unquantized teacher by training scales and biases; defaults are 2048 samples, `--max-seq-length 1025`, `--learning-rate 1e-6`, `--batch-size 4`, data `allenai/tulu-3-sft-mixture` |
| Refine an existing quant | `mlx_lm.dwq --model M --quantized-model Q` | cascades onto a dynamic or mixed quant |
| Activation-aware scaling | `mlx_lm.awq --model M --bits 4 --group-size 64` | only `llama`, `mistral`, `ministral3`, `mistral3`, `llava`, `qwen2`, `qwen3`, `gemma3_text`, `gemma3`, `deepseek_v2`; others raise `AWQ support for <type> models NYI.` |
| Error-minimizing rounding | `mlx_lm.gptq --model M --bits 4` | layers outside the transformer blocks (embeddings, head) get `--fallback-bits 6` |

DWQ tips from upstream: it helps most going down to 2–4 bits and does little for 6–8 bits. `--group-size 32` doubles the trainable scales and often helps. Lower the learning rate if the loss oscillates. Distilling from an 8-bit teacher saves memory with little loss.

`mlx_lm.convert` refuses an existing output directory (`Cannot save to the path ... as it already exists.`), and `--quant-predicate` with a non-affine `--q-mode` raises `Quant predicates only support 'affine' quantization.`

## nvfp4 and mxfp8 specifics

- **Global scale.** `quantize_model(..., mode="nvfp4", global_scale=True)` gives each expert of a `SwitchLinear` a float32 tensor scale. Only switch layers accept it. `gather_qmm` requires float32:

```
[gather_qmm] Global scale must have dtype float32 but got bfloat16.
```

  `mlx_lm.convert --dtype` in mlx-lm 0.32.2 casts the global scale along with everything else and produces exactly this error at inference. The fix (keep any `*global_scale` key in float32) landed after 0.32.2. On 0.32.2, cast with your own `tree_map_with_path` that skips those keys.

- **Activation quantization.** `to_quantized(mode="nvfp4" | "mxfp8", quantize_input=True)` returns a `QQLinear` that quantizes inputs too. Affine raises `Quantized activations are only supported for 'nvfp4' and 'mxfp8' modes, got affine.` A layer with a bias raises `QQLinear does not support bias yet.` mlx-lm enables it from `"quantize_activations": true` in `config.json`, or `mlx_lm.generate -qa`. Measure it against weight-only quantization before adopting it (`mlx-verify`).
- Packed checkpoints load natively. mxfp4, mxfp8, ModelOpt NVFP4, compressed-tensors (unsupported formats raise rather than silently becoming 4-bit affine), and AutoAWQ/GPTQ packed weights are mapped to MLX modes by `load`. Do not dequantize them in `sanitize`.

## Saved dtypes

- `--dtype` casts only floating parameters. It skips any key that the model's `cast_predicate` rejects, so router correction biases stay float32.
- Keep `--dtype` equal to the checkpoint's `torch_dtype`. Casting bf16 weights to float16 caps activations at 65504, which bf16's range does not, and an overflow surfaces as `inf` or `nan` logits.
- `mlx_lm.convert --model Q -d` dequantizes back to float and drops the `quantization` keys. Use it before full fine-tuning or before re-quantizing with a different recipe. `convert -q` on an already-quantized model leaves the existing quantized layers alone and only quantizes the float ones.

## GGUF

- **Reading.** `mx.load("x.gguf")` keeps `Q4_0`, `Q4_1` and `Q8_0` tensors as affine group-32 `(weight, scales, biases)` triples. Every other GGML type is dequantized to float16 at load where the bundled gguflib can decode it, so a Q4_K file grows about 3.5 times in memory. A type it cannot decode fails with `[load_gguf] gguf_tensor_to_f16 failed`. mlx-lm cannot build a model from a GGUF file. Convert from the original safetensors instead.
- **Writing.** `mlx_lm.fuse --model M --adapter-path A --export-gguf [--gguf-path out.gguf]` supports only `llama`, `mistral` and `mixtral`, writes bf16 weights as float16 and norms as float32, and on a quantized base raises `Conversion of quantized models is not yet supported.` Add `--dequantize` for a quantized base. `mx.save_gguf` writes only the arrays and metadata you pass. It does not add the llama.cpp architecture keys, so its output is not a llama.cpp model.

## Publishing to mlx-community

```bash
mlx_lm.convert --model org/Model -q --q-bits 4 --upload-repo mlx-community/Model-4bit
mlx_lm.upload --path mlx_model --upload-repo you/Model-4bit     # for an existing directory
```

- `upload_to_hub` calls `create_repo(exist_ok=True)` with no `private` flag, so a new repo is **public**. For a private repo, create it first with `hf repos create you/Model-4bit --private`.
- `save` writes a model card with `library_name: mlx`, the `mlx` tag and `base_model` set to the source repo. Keep `base_model` correct, since the Hub links lineage through it.
- Name the repo after the quantization people will look for: `-4bit`, `-8bit`, `-mxfp4`, `-4bit-DWQ`, `-mixed_4_6`. Before publishing, check the printed bits per weight and run one `mlx_lm.generate` on the converted directory, not on the source.
