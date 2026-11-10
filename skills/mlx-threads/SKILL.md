---
name: mlx-threads
description: MLX threads and streams — per-thread default streams, what is global and what is per thread, the "no Stream in current thread" error and its causes, new_stream vs new_thread_local_stream vs new_thread_unsafe_stream, handing arrays between threads, loader and worker-pool patterns, per-thread RNG seeding and explicit keys, per-thread compile caches, clear_streams, CPU/GPU overlap with stream=, the GIL and free-threaded Python. Load before touching MLX from more than one Python thread, or when splitting work across the CPU and GPU.
---

# MLX threads and streams

Written against **mlx 0.32.4**. Per-thread default streams arrived in 0.31.2 and were extended to the CPU in 0.32.0, so threading code written before 0.31.2 behaves differently now.

## What is global and what is per thread

| State | Scope | Consequence |
| --- | --- | --- |
| default device (`mx.set_default_device`) | **process** | a worker calling `set_default_device(mx.cpu)` moves the main thread to the CPU too (verified) |
| default stream per device | **thread** | the first op on a new thread creates a new stream: main is `Stream(gpu, 0)`, the first worker gets `Stream(gpu, 1)` |
| command encoders behind a stream | **thread** (the main thread uses a global map) | a stream works only on the thread that created it, except the thread-unsafe kind |
| RNG key (`mx.random.state`) | **thread** | every thread starts from the **same** time-based key |
| `mx.compile` cache | **thread** | the same compiled function traces once per thread |
| transform tracing state (grad, vmap, compile) | **thread** (since 0.32.0) | concurrent `grad`/`compile`/`vmap` on different threads is safe |
| memory limits and buffer cache | **process** | `mlx-memory` |

The "main thread" is whichever thread first touched MLX, not necessarily Python's main thread.

## The cross-thread error

```
RuntimeError: There is no Stream(gpu, 3) in current thread.
```

It means a graph node is scheduled on a stream that belongs to another thread. Three causes, all reproduced on 0.32.4:

1. A stream object made with `mx.new_stream` on one thread is used on another.
2. **A lazy array built on one thread is evaluated on another.** This includes `.tolist()`, `.item()`, `print` and `np.array` on the receiving side, and it fails in both directions: worker to main, and main to worker.
3. A lazy array from another thread is used as an input to new ops, then evaluated here. Even `keys = mx.random.split(key, 4)` on main, indexed inside pool workers, fails until `keys` is evaluated.

An **evaluated** array has no graph, so any thread can read it or use it as an input. The rule: **`mx.eval` before an array crosses a thread boundary**, or hand over numpy.

```python
# Wrong: x is a lazy graph on the loader thread's stream; the consumer's eval raises.
def loader():
    for batch in source:
        q.put(mx.array(batch).astype(mx.bfloat16))

# Right: materialize on the producing thread, then hand it over.
def loader():
    for batch in source:
        x = mx.array(batch).astype(mx.bfloat16)
        mx.eval(x)
        q.put(x)
    q.put(None)
    mx.clear_streams()
```

Handing over numpy works too, and keeps the loader free of MLX state. Copy semantics of `mx.array` vs `mx.asarray` are in `mlx`: a loader that reuses its numpy buffer must not feed `mx.asarray`.

## Stream kinds

| Constructor | Usable from | Use for |
| --- | --- | --- |
| `mx.default_stream(device)` | the current thread; each thread has its own | ordinary code |
| `mx.new_stream(device)` | **only the creating thread** | a second queue on one thread, for overlap |
| `mx.new_thread_local_stream(device)` | any thread; each thread transparently gets its own instance | one handle shared by a worker pool |
| `mx.new_thread_unsafe_stream(device)` | any thread, the same underlying queue | handing a stream across threads. You must serialize all evaluation on it yourself |

```python
tls = mx.new_thread_local_stream(mx.gpu)     # create once, anywhere

def work(seed):
    mx.random.seed(seed)
    with mx.stream(tls):
        y = mx.random.normal((256, 256)) @ mx.random.normal((256, 256))
        mx.eval(y)
        return y.sum().item()

with ThreadPoolExecutor(4) as ex:
    results = list(ex.map(work, range(8)))
```

`with mx.stream(s):` sets the default stream for the current thread only. Re-entering the same context object, and exiting it on another thread, were fixed in 0.32.3. On older versions, use one `with` block per thread.

## Random numbers across threads

The key is thread-local, and every thread's key starts from one process-wide time seed. Without seeding, two workers draw **identical** "random" numbers (verified). `mx.random.seed` on main does not affect workers:

```python
def worker(rank):
    mx.random.seed(base_seed + rank)     # each thread, distinct
    ...
```

For reproducible per-task streams independent of thread scheduling, derive explicit keys on one thread, evaluate them, and pass `key=`:

```python
keys = mx.random.split(mx.random.key(seed), n_tasks)
mx.eval(keys)                                   # required before workers index it
def task(i):
    return mx.random.uniform(shape=(2,), key=keys[i])
```

Since 0.32.1, `mx.random.state` is a process-wide sentinel that resolves to the calling thread's key. A function compiled with `inputs=outputs=mx.random.state` on one thread therefore advances the key of whichever thread calls it. Compile state capture is covered in `mlx-transforms`.

## Compile and transforms on threads

- Each thread keeps its own compile cache. A function compiled on main retraces on its first call from each worker. Expect one trace per thread per input signature.
- `grad`, `vmap` and `compile` may run concurrently on different threads since 0.32.0. Earlier versions shared tracing state and could corrupt results.
- Do not share one **unevaluated** array between threads that both evaluate it. Graph evaluation is not locked. The `new_thread_unsafe_stream` docstring says it plainly: nodes must be evaluated in sequence, and avoiding races is the caller's job.

## clear_streams and thread lifetime

`mx.clear_streams()` destroys the streams the current thread created. It also clears that thread's compile cache and RNG state. Call it as a worker's last MLX action, as upstream's own threaded tests do. The deadlock it could cause while holding the GIL was fixed in 0.32.3.

Stream indices are global and never reused. A thread-per-request server mints new streams forever: a stream created after 50 short-lived threads got index 50. Use a long-lived pool (`ThreadPoolExecutor`) and a `new_thread_local_stream`.

## CPU/GPU overlap with `stream=`

Memory is unified, so no op moves data. Every op takes `stream=` (a `Stream` or a `Device`), and MLX inserts cross-stream and cross-device dependencies automatically:

```python
x = mx.matmul(a, b, stream=mx.gpu)         # compute-dense: GPU
y = b
for _ in range(500):
    y = mx.exp(y, stream=mx.cpu)           # tiny sequential ops: CPU, runs concurrently
mx.eval(x, y)
```

The official docs measure this split at about half the all-GPU time on an M1 Max. Measure your own case (`mlx-verify` owns the method).

- Long chains of tiny ops are launch-bound on the GPU. Big matmuls, convolutions and attention belong on the GPU.
- Each CPU stream is one worker thread, so independent ops on one CPU stream run one after another. For independent chains, create one `mx.new_stream(mx.cpu)` per chain on the issuing thread.
- `float64` and the `mx.linalg` decompositions must run on a CPU stream (see `mlx-transforms`).
- An error inside a CPU-stream op never surfaces at the call site, and **a GPU consumer swallows it**. On 0.32.4, `mx.linalg.inv(singular, stream=mx.cpu) + 1` on the GPU returned `[[1, 1], [1, 1]]` with no exception. The stored error (`[Inverse::eval_cpu] LU factorization failed with error code 1`) appeared only at the next `mx.synchronize(mx.default_stream(mx.cpu))`. Evaluating the CPU result directly, or a CPU-stream consumer of it, does raise. After CPU-stream work that can fail, call `mx.eval` on the CPU output, or synchronize the CPU stream, before a GPU op uses it.
- `mx.synchronize()` waits for the current thread's default stream; `mx.synchronize(s)` waits for one stream.

## The GIL and free-threaded Python

- `mx.eval`, `mx.async_eval`, `mx.synchronize`, `mx.clear_streams` and the eval inside numpy conversion all release the GIL. A Python thread doing I/O or tokenization runs while another thread waits on the GPU.
- Graph building (each op call) holds the GIL. Threads do not parallelize the Python cost of constructing graphs. Compile the hot function (`mlx-transforms`), or batch the work into larger ops.
- Wheels exist for free-threaded CPython 3.13t and 3.14t (since 0.32.1) and 3.15t (since 0.32.4). Every per-thread rule above still applies, and the lack of locking around a shared unevaluated graph becomes a real data race rather than a GIL-serialized one.
