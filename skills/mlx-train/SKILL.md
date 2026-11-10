---
name: mlx-train
description: Training and fine-tuning with MLX — the compiled train-step template with model, optimizer and RNG state, eval cadence and loss reporting, masked fp32 loss, parameter and optimizer dtypes, gradient accumulation and clipping, schedules with warmup, checkpoint and resume including optimizer state, freeze and unfreeze, LoRA, DoRA, QLoRA and full fine-tune, mlx_lm.lora flags, YAML and dataset formats, prompt masking, gradient checkpointing, fuse and dequantize, data parallel with average_gradients, FSDP with fully_shard. Load when writing a training loop or running or debugging mlx_lm.lora.
---

# Training and fine-tuning

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. Compile mechanics (cache keys, retrace triggers, why state capture is needed) belong to `mlx-transforms`. Memory limits and `clear_cache` semantics belong to `mlx-memory`. Launching and backends belong to `mlx-distributed`.

## The train step

```python
from functools import partial
import mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from mlx.utils import tree_map

def loss_fn(model, x, y, mask):
    logits = model(x)
    ce = nn.losses.cross_entropy(logits, y) * mask
    return ce.astype(mx.float32).sum() / mask.sum()

loss_and_grad = nn.value_and_grad(model, loss_fn)     # gradients of trainable params only
state = [model.state, opt.state, mx.random.state]    # RNG too, or dropout repeats its mask

@partial(mx.compile, inputs=state, outputs=state)
def grad_step(x, y, mask):
    return loss_and_grad(model, x, y, mask)

@partial(mx.compile, inputs=state, outputs=state)
def apply(grads):
    grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
    opt.update(model, grads)

model.train()
acc, total = None, mx.array(0.0)
for it, (x, y, mask) in enumerate(batches, 1):
    loss, g = grad_step(x, y, mask)
    acc = g if acc is None else tree_map(mx.add, acc, g)
    total = total + loss
    if it % ACCUM == 0:
        apply(tree_map(lambda a: a / ACCUM, acc))
        acc = None
    mx.eval(state, total, acc)
    if it % REPORT == 0:
        print(it, total.item() / REPORT, opt.learning_rate.item())
        total = mx.array(0.0)
```

- `nn.value_and_grad(model, fn)` with `fn(model, *batch)`. A bare `mx.value_and_grad(fn)(model, ...)` also runs, because a `Module` is a dict, but it differentiates **every** array including frozen ones. `opt.update` then changes the frozen base weights, and a LoRA run silently becomes a full fine-tune.
- `mx.eval(state, ...)` **every step** is correct in training, unlike decode. The step's output is the next step's input, and without the eval the graph spans iterations.
- Keep the loss as an array and accumulate it. Call `.item()` only at report time. A `.item()` per step stalls the queue (`mlx`).
- Sum the masked loss in float32 and divide by the number of real tokens, not by `B * L`. mlx-lm's `default_loss` does this with `(offset, length)` pairs per row.
- Splitting the step into `grad_step` and `apply` keeps two compiled graphs. One compiled function that takes a Python `do_update` flag, or a `prev_grad` that is sometimes `None`, traces once per variant. mlx-lm's trainer does this and gets two traces.
- `model.train()` enables dropout. Evaluation code calls `model.eval()` and must switch back afterwards; mlx-lm's trainer calls `model.train()` after every validation pass.

### Dtypes

MLX has no automatic mixed precision. The optimizer state takes the parameter's dtype, so a bf16 model trained in full gets bf16 Adam moments:

| Setup | Trainable params | Adam state | Recommendation |
| --- | --- | --- | --- |
| LoRA/DoRA on a bf16 or quantized base | float32 (`lora_a`, `lora_b` are created in fp32) | float32 | fine as is; outputs are cast back to `x.dtype` |
| Full fine-tune with `model.set_dtype(mx.bfloat16)` | bf16 | bf16 | works, but an update smaller than bf16's spacing at that weight is lost; keep a float32 copy of the weights for the update if loss stalls |
| Full fine-tune in float32 | fp32 | fp32 | twice the memory; the safe default for small models |

### Schedules

```python
lr = optim.join_schedules([optim.linear_schedule(0, 1e-3, 100), optim.cosine_decay(1e-3, 900)], [100])
opt = optim.AdamW(learning_rate=lr, weight_decay=0.01)
```

The schedule is evaluated from `opt.state["step"]`, so it resumes correctly only when the optimizer state is restored (next section).

### Checkpoint and resume

```python
from mlx.utils import tree_flatten, tree_unflatten
mx.save_safetensors("model.safetensors", dict(tree_flatten(model.trainable_parameters())))
mx.save_safetensors("opt.safetensors", dict(tree_flatten(opt.state)))

model.load_weights("model.safetensors", strict=False)     # trainable subset only
opt.state = tree_unflatten(list(mx.load("opt.safetensors").items()))
```

`mlx_lm.lora --resume-adapter-file` restores **only** the adapter weights, and does it with `load_weights(strict=False)`. Adam moments, the step counter and the schedule restart from zero, so a warmup runs again. A file whose key names do not match loads nothing and raises nothing. For an exact resume, save and restore `opt.state` yourself.

## Choosing a fine-tune type

| Type | What trains | Memory | Use when |
| --- | --- | --- | --- |
| `lora` (default) | low-rank `lora_a`/`lora_b` on the linears of the last `num_layers` blocks | lowest | most instruction or format tuning |
| `lora` on a quantized base (QLoRA) | same; the base stays packed | lowest weight memory | the bf16 base does not fit |
| `dora` | LoRA plus a magnitude vector `m` | slightly more | LoRA underfits; not supported on MoE experts (`SwitchLinear doesn't support DoRA yet.`) |
| `full` | every parameter of the last `num_layers` blocks | highest | domain shift; **needs a float base** |

Full fine-tuning a quantized model fails at the backward pass:

```
RuntimeError: [QuantizedMatmul::vjp] no gradient wrt the quantized weights.
```

`unfreeze()` on a `QuantizedLinear` unfreezes its packed `weight` too. Dequantize first (`mlx_lm.convert --model Q -d --mlx-path base-bf16`), or train only scales and biases the way DWQ does: `m.unfreeze(keys=["scales", "biases"], recurse=False)` on each quantized module.

## LoRA by hand

```python
from mlx_lm.tuner.utils import linear_to_lora_layers
model.freeze()
linear_to_lora_layers(model, num_layers=16, config={"rank": 8, "scale": 20.0, "dropout": 0.0})
```

- The key is `scale`, not `alpha`. Passing `alpha` raises `KeyError: "LoRA config is missing required key 'scale'. Found 'alpha' instead — mlx-lm's LoRA config uses 'scale', not 'alpha' ..."`. To convert a PEFT recipe, use `scale = lora_alpha / r`.
- Without `keys`, every `Linear`, `QuantizedLinear`, `SwitchLinear` and `Embedding` inside the last `num_layers` blocks is wrapped (q/k/v/o/gate/up/down for a llama). `keys` are paths relative to a block (`"self_attn.q_proj"`). Full model paths (`"lm_head"`) are matched against the whole model.
- Freeze **before** wrapping. `linear_to_lora_layers` adds trainable adapters but does not freeze the base. Check with `sum(v.size for _, v in tree_flatten(model.trainable_parameters()))`.
- MoE blocks need `mx.stop_gradient` on the routing indices or the backward fails (`mlx-port`).

## mlx_lm.lora

```bash
mlx_lm.lora --model M --train --data ./data --fine-tune-type lora --num-layers 16 \
  --batch-size 4 --iters 1000 --learning-rate 1e-5 --mask-prompt --grad-checkpoint \
  --adapter-path adapters --save-every 100 --steps-per-eval 200
mlx_lm.lora -c config.yaml        # same keys in YAML, plus lora_parameters and lr_schedule
mlx_lm.lora --model M --adapter-path adapters --data ./data --test
```

```yaml
lora_parameters: {keys: ["self_attn.q_proj", "self_attn.v_proj"], rank: 8, scale: 20.0, dropout: 0.0}
lr_schedule: {name: cosine_decay, warmup: 100, warmup_init: 1e-7, arguments: [1e-5, 1000, 1e-7]}
```

- `lr_schedule.name` is any function in `mlx.optimizers.schedulers`. `arguments[0]` is the peak LR the warmup ramps to.
- Defaults: `batch_size 4`, `iters 1000`, `learning_rate 1e-5`, `num_layers 16`, `max_seq_length 2048`, `lora_parameters {rank 8, scale 20.0, dropout 0.0}`, optimizer `adam` (also `adamw`, `muon`, `sgd`, `adafactor`).
- `--clear-cache-threshold` is parsed but **not passed to the trainer in mlx-lm 0.32.2**. That was fixed after 0.32.2. On 0.32.2 the trainer clears the cache whenever any cache memory is held.
- `--grad-checkpoint` replaces `__call__` on the **class** of `model.layers[0]`, so every block of that class is checkpointed, including blocks in other model instances in the same process. A hybrid model with several block classes gets checkpointing only on the first class.
- Sequences longer than `max_seq_length` are truncated with a warning, not dropped. Batches are sorted by length.

Outputs go to `--adapter-path`: `adapter_config.json`, `adapters.safetensors`, and a checkpoint `0000100_adapters.safetensors` every `--save-every` iterations. `mlx_lm.load(path, adapter_path=...)` and `load_adapters` validate tensor names and shapes and raise `Adapter at ... contains incompatible tensors` on a mismatch.

### Data

`--data` is a directory with `train.jsonl`, `valid.jsonl` and `test.jsonl`, or a Hub dataset id. Each file's format is chosen from **its first line only**:

| First record has | Dataset | Tokenized with | `--mask-prompt` |
| --- | --- | --- | --- |
| `prompt` and `completion` | completions | the chat template as user/assistant turns | masks the user turn |
| `messages` (optional `tools`) | chat | the chat template | masks everything before the last assistant message |
| `text` | text | plain `encode`, EOS appended if missing | raises `Prompt masking not supported for text dataset.` |

Anything else raises `Unsupported data format, check the supported formats here:`. Completions and chat need a tokenizer with a chat template. A base model without one fails with `Cannot use chat template functions because tokenizer.chat_template is not set ...`, so use `text` records for base models. Field names can be changed with `prompt_feature`, `completion_feature`, `chat_feature` and `text_feature` in the YAML. `hf_dataset: {path, train_split, valid_split, prompt_feature, ...}` loads a Hub dataset with slicing. Training requires `train.jsonl`. A missing `valid.jsonl` only prints a warning.

## Fuse

```bash
mlx_lm.fuse --model M --adapter-path adapters --save-path fused            # quantized base stays quantized
mlx_lm.fuse --model M --adapter-path adapters --save-path fused --dequantize
```

On a quantized base, `fuse` dequantizes, adds `scale * B^T A^T`, and **re-quantizes** with the base's settings. A small adapter delta can fall below the 4-bit rounding step and disappear. Compare the fused model against the unfused `--adapter-path` run. If it regressed, fuse with `--dequantize` and re-quantize with a recipe or DWQ (`mlx-quant`).

## Data parallel and FSDP

**Data parallel.** Average the gradients inside the compiled step, before the update:

```python
from mlx.nn.utils import average_gradients
grads = average_gradients(grads)          # all_sum / N, bucketed into 32 MiB messages
opt.update(model, grads)
```

- Shard the data by rank: `batch[rank::size]`. mlx-lm requires `batch_size % world_size == 0` and raises otherwise.
- Every rank seeds the model identically (`mx.random.seed(0)` before construction), or the ranks start from different weights and never agree.
- Report the global loss with `mx.distributed.all_sum(loss, stream=mx.cpu).item() / size`. Save on rank 0 only.

**FSDP.** `nn.fully_shard(module, group=None, compute_dtype=None)` (since mlx 0.32.1) keeps 1/N of each parameter per rank. It all-gathers the full weights for the forward pass and reduce-scatters the gradients in the backward pass.

```python
for layer in model.model.layers:
    layer.self_attn = nn.fully_shard(layer.self_attn)
    layer.mlp = nn.fully_shard(layer.mlp)
...
grads, norm = clip_grad_norm_sharded(grads, max_norm=1.0)   # from mlx.nn.utils
opt.update(model, grads)                                      # no average_gradients
```

- Wrap submodules, not whole decoder blocks. The wrapper hides the wrapped module's attributes, so `nn.fully_shard(layer)` on an mlx-lm block fails with `AttributeError: 'FullyShardedModule' object has no attribute 'use_sliding'` when the model reads `layer.use_sliding`.
- Gradients come back already averaged (`sum_scatter / N`). Calling `average_gradients` again divides by N twice.
- Axis 0 of every wrapped parameter must divide by N: `Cannot shard parameter '...' with shape ... across N devices: axis 0 must be divisible by N.`
- Use `clip_grad_norm_sharded`. Plain `clip_grad_norm` sees only the local shard's norm.
- The backward pass needs `sum_scatter`, which ring and MPI do not implement (`[ring] sum_scatter not supported.`), so FSDP needs JACCL or NCCL (`mlx-distributed`). On a single process, `fully_shard` returns the module unchanged.
