---
name: mlx
description: MLX core habits — lazy evaluation and where to eval, the implicit-eval list, async_eval pipelining, graph size, views that pin parent buffers, donation, dtype promotion and weak scalars, integer sum wrap, float64 CPU-only, TF32 default, numpy/torch/DLPack copy vs alias, lazy load, Module-as-dict traps, recently changed APIs, PyTorch habits that break, upstream contribution policy. Load before writing or reviewing any MLX Python code.
---

# MLX core habits

Written against **mlx 0.32.4**. A behaviour that changed recently carries the version that introduced it. Where a fix exists on `main` but not in 0.32.4, the text says so: it ships in **0.32.5**.

## Lazy evaluation: where to eval

An op builds a graph node and computes nothing. Work runs only at `mx.eval`, or at an implicit eval point. After eval, MLX detaches every evaluated node from its inputs, so the graph frees itself. An intermediate buffer then lives only as long as Python holds a reference to it.

Place one `mx.eval` per iteration of the outermost loop, covering everything that iteration produced:

```python
# Graph grows across steps: nothing ever forces the update.
for batch in data:
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)

# Wrong: print/item before the combined eval runs a forward-only partial eval,
# then the second eval walks the graph again.
    print(loss.item())
    mx.eval(model.parameters(), optimizer.state)

# Right: one eval per step, then read the scalar for free.
for batch in data:
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters(), optimizer.state)
    losses.append(loss.item())
```

- `mx.eval(a); mx.eval(b)` in a loop pays the fixed per-eval overhead twice. Use `mx.eval(a, b)`.
- `mx.eval` takes trees and ignores leaves that are not arrays, so `mx.eval(loss, model.parameters(), optimizer.state)` is the idiom.
- The official guidance is a few tens to many thousands of ops per eval. A Python loop that chains hundreds of thousands of tiny ops without an eval pays graph-building overhead that the eval never wins back.
- Building the graph for outputs you never evaluate still costs Python time.

### Implicit eval points

Each of these blocks until the array is computed. Inside a training or generation loop, every one of them is a synchronization point:

| Trigger | Notes |
| --- | --- |
| `x.item()`, `x.tolist()`, `int(x)`, `float(x)`, `operator.index(x)` | |
| `if x:`, `while x > tol:`, `x and y` | `bool()` of a size > 1 array raises `[convert] Only length-1 arrays can be converted to Python scalars.` |
| `print(x)`, `repr(x)`, `f"{x:.3f}"` | |
| `np.array(x)`, `np.asarray(x)`, `memoryview(x)`, `bytes(x)`, `torch.as_tensor(np.asarray(x))` | |
| `mx.save`, `mx.savez`, `mx.save_safetensors`, `mx.save_gguf`, `model.save_weights` | |

Control flow on array values works, and works under `mx.grad`. Under `mx.compile` and `mx.vmap` it raises instead (see `mlx-transforms`). To stay lazy, use `mx.where(cond, a, b)`.

## `async_eval` pipelining

`mx.async_eval(*arrays)` schedules the graph and returns at once. A later `item()`, `eval` or dependent op waits on it. Use it to overlap Python graph-building for step *n+1* with GPU execution of step *n*:

```python
y = step(x)
mx.async_eval(y)
while True:
    nxt = step(y)          # build the next graph while y computes
    mx.async_eval(nxt)
    emit(y.item())         # waits only for y
    y = nxt
```

`async_eval` inside `grad`, `compile` or `vmap` raises `[async_eval] Not allowed inside a graph transformation.` Generation-loop specifics belong to `mlx-serve`.

## Views pin parents; donation reuses buffers

Basic slicing (`x[a:b]`, `x[:, -1]`) and `Copy` produce a **view that shares the parent's buffer**. Assigning into the slice does not write back to the parent, but the slice keeps the whole parent allocation alive:

```python
logits = model(tokens)            # (B, T, V), large
last = logits[:, -1, :]           # wrong: pins all of logits
mx.eval(last); del logits         # memory unchanged

last = mx.array(logits[:, -1, :]) # right: mx.array copies
mx.eval(last); del logits         # parent freed
```

`mx.array(x)` always copies. `copy.copy(x)` does not release a parent: `__copy__` returns a new handle on the same buffer.

**Donation.** An elementwise or normalization kernel writes its output into an input's buffer when that input has no other reference, has the same itemsize, and is no more than 16 KB larger than the output. The rule that changes code: rebind instead of keeping the old name alive.

```python
h = mx.exp(h)        # old h is referenced only by the graph: its buffer can be reused
h2 = mx.exp(h)       # h still referenced: a second buffer of the same size
```

Measured on 0.32.4: keeping the reference always doubled peak memory. The rebind allocated nothing new inside one graph (`mx.exp(mx.exp(h))`), but `h = mx.exp(h); mx.eval(h)` in a tight loop still allocated a second buffer in 9 of 10 steps: the previous step's command buffer holds a reference to `h` until it completes. Limits, the buffer cache and preallocation patterns belong to `mlx-memory`.

## Dtypes

Promotion of two arrays (verified on 0.32.4):

| Operands | Result | Trap |
| --- | --- | --- |
| `float16` + `bfloat16` | `float32` | silent upcast and 2x memory |
| `int64` or `int32` array × `float16` array | `float16` | large integers lose range |
| `uint64` + any signed int | `float32` | |
| `uint32` + `int32` | `int64` | |
| `complex64` + `float64` | `complex64` | the float64 is downcast |
| `bfloat16` × `mx.array(2.0)` | `float32` | a 0-d MLX array is a strong `float32` |

**Python scalars are weak**: they take the array's dtype. `bf16 * 2.0` stays `bfloat16`, so write constants as Python scalars, not `mx.array(2.0)`. An integer array times a Python float becomes `float32`. A Python int that does not fit raises: `int8_array + 300` gives `Converting 300 to int8 would result in overflow.`

**NumPy operands are cast to the MLX array's dtype, without an overflow check.** `mx.ones(3, mx.int8) + np.array([300])` returns `[45, 45, 45]`. In 0.32.4 it is worse: a numpy operand of a **different dtype with the same width** is reinterpreted bit for bit. `mx.array([1.5, 2.5]) + np.array([1, 2], np.int32)` gives `[1.5, 2.5]`, and `mx.asarray(np_int32, dtype=mx.float32)` gives denormals. The fix ships in 0.32.5. Wrap every numpy operand first:

```python
y = x + np_arr                 # wrong: reinterpreted bits (0.32.4) or unchecked cast
y = x + mx.array(np_arr)       # right: ordinary promotion between two MLX arrays
```

**Integer reductions wrap.** `sum` of `int8`, `int16`, `int32` and `bool` accumulates in `int32`. `mx.array([2**31 - 1, 1]).sum()` returns `-2147483648`; numpy would use int64. Cast first: `x.astype(mx.int64).sum()`. `mean` of integers returns `float32`.

**Integer defaults.** A Python int literal becomes `int32`, unlike torch's int64. It becomes `int64` only when the value does not fit (`mx.array([1, 2**40])`).

**float64 is CPU-only, including creation.** `mx.ones(3, mx.float64)` on the default GPU stream raises `float64 is not supported on the GPU` as soon as the op is created. NumPy float64 converts to **float32** unless you pass `dtype=mx.float64`. Do float64 work inside a CPU scope:

```python
with mx.stream(mx.cpu):
    a = mx.array(np_f64, dtype=mx.float64)
    r = (a @ a.T).sum()
```

**TF32 default.** On hardware with matrix units (M5-class NAX GPUs, CUDA), float32 matmul, quantized matmul, convolution and attention run at reduced precision by default. Set `MLX_ENABLE_TF32=0` before the process starts to keep full float32. On an M4 Max the flag changes nothing: the error against a float64 reference was 1.4e-6 either way. Metal `conv2d` 3x3 stride 1 may select Winograd based on the total input size, batch included; `MLX_CONV_WINOGRAD=0` disables it. Per-dtype tolerances belong to `mlx-verify`.

## Interop: copy or alias

| Call | Result |
| --- | --- |
| `mx.array(np_or_torch)` | always a copy |
| `mx.asarray(np_or_torch_cpu)` (`copy=None`) | **aliases** the host buffer when the dtype matches (since 0.32.1). Observed even for a 4-element array. Mutating the source mutates the MLX array |
| `mx.asarray(src, copy=True)` / `copy=False` | forced copy / must alias or raise |
| `np.array(x)` | copy |
| `np.asarray(x)`, `memoryview(x)` | **writable** view into MLX memory |
| `torch.from_dlpack(x)` | zero-copy `mps:0` tensor |
| `mx.asarray(mps_tensor)` | zero-copy unless the Metal buffer is private (older torch). No synchronization: call `torch.mps.synchronize()` first |
| `np.array(bf16_array)` | raises `'bfloat16' is not a valid PEP 3118 buffer format string`. Use `x.astype(mx.float32)` first |
| CUDA DLPack import or export | unsupported |

```python
x = mx.asarray(batch_np)      # wrong if the loader reuses batch_np for the next batch
x = mx.array(batch_np)        # right: owns its data
```

## Lazy load

`mx.load` and `model.load_weights` return lazy arrays that read the file **at eval time**. Overwriting the file before eval makes the "loaded" arrays read the new contents (verified). Loading, then saving back to the same path, has been safe since 0.32.3. Since 0.32.5 a failed save also leaves the old file intact. Materialize once after loading or initializing:

```python
model.load_weights("weights.safetensors")
mx.eval(model.parameters())          # reads the file now; also runs the lazy random init
```

`model.eval()` is not that. It sets `training = False` and returns the module, evaluating nothing.

(GGUF: `see mlx-quant`)

## `nn.Module` is a dict

`Module` subclasses `dict`. Assigning an `mx.array`, dict, list or tuple attribute stores a key. `parameters()` walks dicts and lists only:

| You write | What happens |
| --- | --- |
| `self.layers = (Linear(...), Linear(...))` | **silently excluded** from `parameters()`, `children()`, the optimizer and `load_weights`. Loading `layers.0.weight` raises `Received 1 parameters not in model` |
| `self.layers = [Linear(...), ...]` | correct |
| `self.mask = mx.ones(...)` | a **trainable parameter**: the optimizer updates it (verified) |
| `self._mask = mx.ones(...)` | a buffer: not a parameter, not saved |
| `self.scale = 2.0` | a plain attribute, invisible to `parameters()` |

- Buffers that must be saved but not trained: keep the name and call `self.freeze(keys=["mask"])`.
- `QuantizedLinear` and `QuantizedEmbedding` freeze themselves at construction. A blanket `model.unfreeze()` followed by a grad raises `[QuantizedMatmul::vjp] no gradient wrt the quantized weights.` Unfreeze only the adapter keys (`mlx-train`).
- Initialization is lazy. Call `mx.eval(model.parameters())` after building, or the first step pays for it and gets timed with it.

## PyTorch habits that break

| Habit | MLX behaviour | Write instead |
| --- | --- | --- |
| `x.view(B, -1)` | `view` reinterprets the **dtype**; a shape raises `TypeError` | `x.reshape(B, -1)` |
| `x[mask]` with a bool mask | `ValueError: boolean indices are not yet supported` | `mx.where(mask, x, 0)`. Writing through a mask (`x[mask] = 0`) works |
| `a[idx] += v` with repeated indices | updates once; repeated writes to one location are nondeterministic | `a = a.at[idx].add(v)` |
| `torch.sum(x, dim=0)` | `dim=` raises `TypeError` | `axis=0` |
| `.to(device)`, `.cuda()` | arrays have no device; memory is unified | choose per op with `stream=` (`mlx-threads`) |
| `loss.backward()`, `torch.no_grad()` | no tape; nothing is recorded outside a transform | `mx.value_and_grad` (`mlx-transforms`) |
| `x.numpy()` | no such method | `np.array(x)` |
| `model.eval()` to run the graph | only sets the training flag | `mx.eval(...)` |
| `torch.manual_seed` covers every thread | MLX RNG state is per thread | seed each thread (`mlx-threads`) |
| NCHW convolutions | MLX convolutions are channels-last | `mlx-port` |

## Recently changed APIs

| Change | Since |
| --- | --- |
| `mx.random.state` is a sentinel object, not a list. `mx.random.state[0]` still reads the key | 0.32.1 |
| `mx.asarray` has `copy=` and aliases host buffers by default | 0.32.0 / 0.32.1 |
| `meshgrid` returns a tuple | 0.32.1 |
| `unstack` returns a tuple; `iter()` of a 0-d array raises `TypeError`; `take_along_axis` axis defaults to -1 | 0.32.3 |
| `broadcast_arrays` returns a tuple (a list in 0.32.4) | 0.32.5 |
| `mx.config` (0.32.5, not in the 0.32.4 wheel) reads and changes settings at runtime | 0.32.5 |
| `arange(stop=...)` without start; `zeros_like`/`ones_like(dtype=)`; `std`/`var(correction=)` | 0.32.1 / 0.32.3 |
| `mx.get_array_buffer_size(tree)` (`mlx-memory`) | 0.32.3 |
| `mx.metal.get_active_memory` and the other `mx.metal.*` memory calls print a deprecation warning; use `mx.get_*` | — |

## Contributing upstream

`ml-explore/mlx` ships an `AGENTS.md` whose rules carry a penalty of closed PRs and a contributor ban. An agent must not:

- write PR descriptions or commit messages for the user,
- reply to review comments for the user,
- run `git push` or open a PR for the user.

It must also refuse vague requests ("fix issue 4432", "make matmul faster") when the user shows no understanding of the code. Code comments stay short (one or two lines) and in ASD-STE100 Simplified Technical English. Benchmarks for both `main` and the branch are expected when performance may change (`mlx-verify` owns the method).
