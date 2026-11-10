---
name: mlx-memory
description: MLX memory on Apple silicon — unified memory and what the limits mean, memory-limit defaults and limit semantics, the buffer cache and its reuse window, cache limit and clear_cache cadence, set_wired_limit and its cap, the iogpu.wired_limit_mb sysctl, residency sets, measuring active, peak and cache memory and get_array_buffer_size, preallocation and buffer reuse, deprecated mx.metal.* calls, and a decoder for Insufficient Memory, kIOGPUCommandBufferCallbackErrorTimeout, Resource limit exceeded and maximum buffer size errors. Load before sizing a model or batch for a Mac, when memory grows without bound, or when a run dies with a Metal memory or timeout error.
---

# MLX memory

Written against **mlx 0.32.4** on macOS 26.2+. Numbers are from an M4 Max with 128 GB (`max_recommended_working_set_size` 115448725504). Views that pin parent buffers and donation belong to `mlx`. KV cache layout belongs to `mlx-serve`. Per-thread streams belong to `mlx-threads`. Memory knobs are process-wide.

## The numbers that matter

```python
info = mx.device_info()                 # was mx.metal.device_info()
info["memory_size"]                     # physical RAM: 137438953472
info["max_recommended_working_set_size"]  # what macOS lets the GPU use well: 115448725504 (about 84% of RAM)
info["max_buffer_length"]               # largest single MTLBuffer: 86586540032
info["resource_limit"]                  # live buffer count limit: 499000
```

| Knob | Default | What it does |
| --- | --- | --- |
| `mx.set_memory_limit(n)` | min(1.5 × max recommended working set, 0.95 × RAM). Here 130567005798 (121.6 GiB, the 0.95 × RAM bound) | eval **waits** for in-flight work to finish when active memory exceeds it. Not a cap |
| `mx.set_cache_limit(n)` | equal to the memory limit | freed buffers kept for reuse up to `n`; `0` disables the cache |
| `mx.set_wired_limit(n)` | `0` (nothing wired) | keeps up to `n` bytes GPU-resident through residency sets; capped at max recommended working set |
| `MLX_RESIDENCY_SET_MAX_PCT` | 5 | size of each residency set, as a percentage of the working set (since 0.32.1); does not change the total |

All four setters return the previous value, so a context manager can restore it:

```python
old = mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
try:
    run()
finally:
    mx.synchronize()
    mx.set_wired_limit(old)
```

## The memory limit throttles; it does not cap

```python
mx.set_memory_limit(1 << 30)            # 1 GB
x = mx.zeros((1 << 30,))                # 4 GB of float32
mx.eval(x)                              # succeeded; active memory 4.0 GB
```

When active memory is over the limit and work is in flight, eval commits open command buffers and waits for them before scheduling more. That lowers the peak of a large graph; it never refuses an allocation. On iOS it does not prevent jetsam (`see mlx-swift-app`). To bound memory, bound the work: smaller batches, chunked prefill, `mx.checkpoint` (`see mlx-transforms`), eval more often (`see mlx`).

## The buffer cache

Freed buffers go to a cache instead of back to the OS. That is why memory in Activity Monitor stays high after `del`: after deleting the 4 GB array above, `mx.get_cache_memory()` reported 4.0 GB.

- A cached buffer is reused only for a request in `[size, min(2 × size, size + 2 pages))`. Shapes that change every step (growing sequences, ragged batches) miss the cache and allocate fresh, so the cache grows until it reaches its limit.
- Over the cache limit, the next allocation frees cached buffers first.

| Workload | Cache setting |
| --- | --- |
| Training or inference with fixed shapes | leave the default; the cache makes steady state allocation-free |
| LLM generation with growing context | `mx.clear_cache()` every 256 decode tokens and after each prefill chunk (what `mlx_lm.generate` does) |
| Training with variable lengths | clear only above a threshold: `if mx.get_cache_memory() > threshold: mx.clear_cache()` (mlx-lm trainer) |
| Apps sharing the machine (chat UI, phone) | `mx.set_cache_limit(20 << 20)` once at startup |
| Debugging "where is my memory" | `mx.set_cache_limit(0)`; then `get_cache_memory()` stays 0 |

`mx.clear_cache()` after every op or every step costs a fresh allocation and residency change for each buffer. Do it on a cadence, not in the inner loop.

## Wired memory and the sysctl

Without wiring, macOS may page GPU buffers out of residency between steps. Large models then stall, or fail with `Insufficient Memory`. mlx-lm wraps generation in a `wired_limit` context that sets the wired limit to `max_recommended_working_set_size`. It prints this when the model is over 0.9 of that size:

```
[WARNING] Generating with a model that requires {model_mb} MB which is close to the maximum recommended size of {max_rec_mb} MB. This can be slow. See the documentation for possible work-arounds: ...
```

- `set_wired_limit` above the working set raises `ValueError: [metal::set_wired_limit] Setting a wired limit larger than the maximum working set size is not allowed.`
- To raise the working set itself, set the macOS limit (needs `sudo`, resets on reboot, and 0 means the OS default):

  ```
  sysctl iogpu.wired_limit_mb                  # 0 = macOS default
  sudo sysctl iogpu.wired_limit_mb=N           # N: above the model size in MB, below total RAM in MB
  ```

  Leave several GB for the OS; wiring nearly all RAM freezes the machine under pressure. Restart the Python process after changing it: MLX reads the working set once, when its allocator starts.
- `set_wired_limit` only works on macOS 15 and later.
- Wired memory is split over residency sets of 5% of the working set each (minimum 64 MiB). A set that loses residency under pressure costs only its own buffers to restore. `MLX_RESIDENCY_SET_MAX_PCT=0` (or 100) uses one set.
- `MLX_RESIDENCY_DEBUG=1` logs each set as it is created. `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=N` re-requests residency every N ms, for servers that idle and then stall on the first request, because macOS can drop residency after GPU idle.
- Like most `MLX_*` variables, these are read once per process (`see mlx-hot-path`).

## Measuring

| Question | Call |
| --- | --- |
| Live MLX buffers now | `mx.get_active_memory()` |
| High-water mark since last reset | `mx.reset_peak_memory()`, run, then `mx.get_peak_memory()` |
| Freed but held | `mx.get_cache_memory()` |
| Bytes behind specific arrays or a model (since 0.32.3) | `mx.get_array_buffer_size(model.parameters())` |

`get_array_buffer_size` counts allocator sizes of unique buffers, not `nbytes`:

| Input | `nbytes` | `get_array_buffer_size` |
| --- | --- | --- |
| 10 int32 | 40 | 40 (heap allocation) |
| 5000 float32 | 20000 | 32768 (page rounded) |
| `y[:10]` of a 4 MB array | 40 | 4194304 (the view pins the parent buffer) |
| `[y, y[:10]]` | | 4194304 (each buffer counted once) |
| `nn.Linear(1000, 1000)` parameters | 4004000 | 4018080 |
| unevaluated array | | `[get_array_buffer_size] Arrays must be evaluated before querying buffer size.` |

Non-array leaves in the tree are ignored. The result excludes the cache, graph objects and Python memory.

- Peak memory is the number to size against: a step's peak is weights plus activations plus temporaries, all at once.
- `mx.get_active_memory()` counts MLX buffers only. It excludes the cache, PyTorch and numpy memory, and model files that are memory-mapped but not loaded.
- `mx.eval` before reading any counter; lazy arrays have no buffer yet.

## Deprecated `mx.metal.*` calls

Each still works and prints one warning to stderr, once per process, for example `mx.metal.get_active_memory is deprecated and will be removed in a future version. Use mx.get_active_memory instead.`

| Deprecated | Use |
| --- | --- |
| `mx.metal.get_active_memory`, `get_peak_memory`, `reset_peak_memory`, `get_cache_memory` | `mx.` same name |
| `mx.metal.set_memory_limit`, `set_cache_limit`, `set_wired_limit`, `clear_cache` | `mx.` same name |
| `mx.metal.device_info()` | `mx.device_info()` |

`mx.metal.is_available()`, `mx.metal.start_capture` and `stop_capture` are not deprecated. There is no `mx.metal.get_memory_limit`; use `mx.get_memory_limit()`.

## Preallocate and reuse

```python
# Wrong: a concatenate per token allocates a new, slightly larger buffer every step; none fit the cache window.
k_cache = mx.concatenate([k_cache, k_new], axis=2)

# Right: grow in fixed steps and write in place (mlx-lm KVCache uses step = 256).
if offset + n > k_cache.shape[2]:
    k_cache = mx.concatenate([k_cache, mx.zeros((B, H, 256, D), k_cache.dtype)], axis=2)
k_cache[..., offset:offset + n, :] = k_new
```

- Keep shapes in a small set of buckets (pad batches to multiples of 8 or 64). Fixed shapes hit the cache and avoid `mx.compile` retraces.
- Drop references before the next step allocates. `del` the old logits, or rebind a name, so the buffer can be reused or donated (`see mlx`).

## Error decoder

| Message | Meaning | Fix |
| --- | --- | --- |
| `[METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)` | the GPU could not make the command buffer's buffers resident. Upstream reports: LoRA and fine-tuning near RAM size, large models without wiring | smaller batch or sequence; `set_wired_limit` to the working set; raise `iogpu.wired_limit_mb`; close other GPU apps; `mx.clear_cache()` between phases |
| `[METAL] Command buffer execution failed: Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)` | the GPU watchdog killed a command buffer. Upstream reports put the window near 5 s and count time spent waiting on fences, so it shows up in distributed runs where one rank waits on another, and in MoE decode on 16 GB machines under memory pressure | lower the per-buffer work (`MLX_MAX_OPS_PER_BUFFER`); keep ranks in step (`see mlx-distributed`); reduce memory pressure. A single 30 s compute dispatch on an otherwise idle M4 Max did not trip it, so the timeout is mostly a stall, not long math |
| `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted` | an iOS or iPadOS app submitted GPU work while backgrounded | stop generation on the background notification (`see mlx-swift-app`) |
| `[metal::malloc] Resource limit (499000) exceeded.` | about 499k live MTLBuffers. Measured: reached at 498999 live arrays, with only 1.9 GB in use | stop holding many small arrays (per-token lists, per-step metrics); stack them; eval and drop. Upstream: a `mx.compile` cache retaining buffers for every distinct shape |
| `[metal::malloc] Attempting to allocate 88000000000 bytes which is greater than the maximum allowed buffer size of 86586540032 bytes.` | one array over `max_buffer_length` | shard the tensor (per layer, per expert) |
| `OverflowError: Integer value 21646636032 is outside the supported range [-2147483648, 2147483647].` | one dimension over `INT32_MAX` (shapes are int32) | split the dimension |
| `ValueError: [metal::set_wired_limit] Setting a wired limit larger than the maximum working set size is not allowed.` | wired limit above the working set | raise the sysctl first, then restart |
| process killed, no Python error; `jetsam` or "Killed: 9" | the OS killed the process for memory; MLX's limit is not a cap | lower peak memory |

`iogpu.rsrc_limit` overrides the resource limit when the sysctl exists. It does not exist on macOS 27.3 (`unknown oid`), so 499000 is the fallback.

## Anti-patterns

| Anti-pattern | Fix |
| --- | --- |
| `set_memory_limit` to keep a process inside a budget | bound the work; the limit only throttles |
| Reading Activity Monitor after `del` | `get_active_memory()` vs `get_cache_memory()` |
| `mx.clear_cache()` every op | a cadence: every 256 tokens, or above a threshold |
| A concatenate per token for caches | step-allocated buffers with slice writes |
| `set_wired_limit` above the working set | the sysctl, then restart |
| `sudo sysctl iogpu.wired_limit_mb` set to all of RAM | leave the OS several GB |
| A Python list of 500k small arrays | stack, or keep numpy scalars |
| `mx.metal.get_peak_memory()` in new code | `mx.get_peak_memory()` |
| Sizing from `nbytes` | `get_array_buffer_size` or peak memory |
