---
name: mlx-verify
description: Verifying MLX code — references (numpy float64, torch, CPU vs GPU, transformers), measured per-dtype tolerances and a scale-aware comparison, precision switches for tests, adversarial shapes, views and special values, gradient checks, determinism, benchmark method, model-port checks (logits, greedy tokens, cache consistency, the mlx-lm runner), MLX's own test layout, CI shape. Load when deciding how to test, check, or benchmark an MLX change.
---

# Verifying MLX code

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. Profiling, finding *where* time goes, belongs to `mlx-hot-path`; this skill owns whether a result is right and whether a number moved. Lazy-evaluation rules are in `mlx`.

## What a passing test must establish

- **Against an independent reference**, never against MLX itself run the same way. A round-trip test (`dequantize(quantize(w))` close to `w`) proves reversibility, not correctness.
- **Shape and dtype**, asserted separately from values. A silent upcast from bf16 to float32 passes every value check and doubles memory.
- **Evaluated.** `mx.allclose` returns a lazy array; `assert mx.allclose(a, b)` works because `assert` calls `__bool__`, which evaluates. A test that only builds the graph has tested nothing.
- **On the device that ships.** A GPU suite on a runner without Metal silently runs on the CPU (see CI).

## References

| Reference | Use when | Watch for |
| --- | --- | --- |
| numpy in float64 | default for array ops | convert with `np.array(x.astype(mx.float32))`; `np.array(bf16_array)` raises |
| torch | bf16 semantics, autograd, ops numpy lacks (SDPA, conv variants) | layouts differ: torch conv is NCHW with `(O, I, H, W)` weights, MLX is NHWC with `(O, H, W, I)` |
| the same MLX op on `mx.cpu` | differential check of a new GPU kernel, or ops with no numpy twin | CPU and GPU share the graph code, so this catches backend bugs, not spec misreadings |
| MLX float64 on `mx.cpu` | finite-difference gradients | float64 is CPU-only (`ValueError: float64 is not supported on the GPU`); in 0.32.4 CPU `exp`, `sin`, `cos` and `erf` compute float64 at **float32 precision** (relative error ~1e-7), fixed after 0.32.4 (#4789) |
| transformers on identical weights | model ports | see "Model ports" |

## Tolerances

Compare on a **scale-aware** metric. Elementwise relative error is meaningless where the reference crosses zero: for a bf16 row sum the 99th-percentile elementwise relative error measured here was 0.24 while the normalized error was 5e-3.

```python
import numpy as np, mlx.core as mx

TOL = {mx.float32: 1e-5, mx.float16: 2e-3, mx.bfloat16: 2e-2}

def assert_close(out, ref, dtype=None):
    """out: mx.array; ref: numpy array computed in float64 from the same (rounded) inputs."""
    dtype = dtype or out.dtype
    o = np.array(out.astype(mx.float32)).astype(np.float64)
    assert o.shape == ref.shape, (o.shape, ref.shape)
    err = np.abs(o - ref).max() / max(np.abs(ref).max(), 1e-30)
    assert err <= TOL[dtype], f"{dtype}: max|out-ref|/max|ref| = {err:.2e} > {TOL[dtype]:.0e}"
```

Measured on an M4 Max, `MLX_ENABLE_TF32=0`, `max|out − ref| / max|ref|` against float64 numpy on the same rounded inputs:

| Op | float32 | float16 | bfloat16 |
| --- | --- | --- | --- |
| matmul, K=1024 | 6e-7 | 2.5e-4 | 3.1e-3 |
| row sum, n=1024 | 1.1e-7 | 6.4e-4 | 5.2e-3 |
| exp | 7e-8 | 2.3e-4 | 1.5e-3 |
| softmax | 6e-8 | 3.2e-4 | 5.0e-3 |
| `fast.rms_norm` | 9e-8 | 4.5e-4 | 3.6e-3 |
| `fast.scaled_dot_product_attention`, D=64 | 5.7e-7 | 2.5e-4 | 2.4e-3 |

The half-precision numbers are the output rounding itself (half an ulp is ~4.9e-4 for float16 and ~3.9e-3 for bfloat16), so a kernel that does better than this table is not possible and one that does 10x worse is a bug. The `TOL` values are roughly 4x the worst case measured, and they grow with reduction length: re-measure for K or n far above 1024 rather than loosening blindly.

`mx.allclose` defaults are `rtol=1e-5, atol=1e-8`, tight enough for float32 elementwise ops and wrong for anything in half precision.

## Precision switches in test environments

- Set `MLX_ENABLE_TF32=0` and `MLX_CONV_WINOGRAD=0` in CI before the first op (`see mlx`).
- `mx.random.normal` is deterministic for a key on one machine but **not bit-exact across GPU architectures**. Stored golden outputs need stored inputs: generate them with `np.random.default_rng(seed)` and save `.npy`, never "reseed MLX and regenerate".

## Adversarial inputs

The bugs that MLX itself fixes are int32 index overflow, negative strides, partial SIMD groups and broadcast inputs. Run every new op or kernel over:

```python
import itertools, mlx.core as mx

SIZES = [1, 7, 31, 32, 33, 1025]          # below 8, around the SIMD width of 32, odd remainder
DTYPES = [mx.float32, mx.float16, mx.bfloat16]

def variants(rows, cols, dtype=mx.float32):
    r = lambda *s: mx.random.normal(s).astype(dtype)
    yield "contiguous", r(rows, cols)
    yield "transposed", r(cols, rows).T
    yield "broadcast",  mx.broadcast_to(r(1, cols), (rows, cols))   # stride 0
    yield "reversed",   r(rows, cols)[::-1]                          # negative stride
    yield "strided",    r(rows, 2 * cols)[:, ::2]
    yield "offset",     r(rows + 1, cols)[1:]                         # non-zero buffer offset

for (rows, cols), dt in itertools.product(itertools.product([1, 3, 33], SIZES), DTYPES):
    for name, x in variants(rows, cols, dt):
        ...  # compare fn(x) against the reference on np.array(x.astype(mx.float32))
```

- A size **below 8** matters beyond the SIMD width: `mx.fast.metal_kernel` binds small inputs to the `constant` address space instead of `device`, so a helper that compiles for a large test fails on a small one (`mlx-kernel`).
- Add one case past 2^31 elements for anything that indexes, if the machine has the memory. 2026 fixes alone cover concatenate, reductions, convolution and gather/scatter offsets past 32-bit sizes.
- For `shapeless=True` functions, test with a second shape where dimensions differ and where two dimensions swap sizes, because a traced reshape replays its trace-time shape.
- Special values: `-inf` in softmax inputs (masking), a row of all `-inf` (MLX returns `nan`, as numpy does), float16 magnitudes past 65504 (`mx.full((4096,), 30., dtype=mx.float16).sum()` is `inf`), and empty arrays (`(0, 8)` is valid and must keep its shape).

## Gradients

Cheapest correct check: torch autograd on the same float32 inputs. It agreed with `mx.grad` to 1.2e-7 here.

Finite differences need float64 to mean anything, which means the CPU:

```python
def check_grad(f, x, eps=1e-6, tol=1e-6):
    with mx.stream(mx.cpu):
        x = x.astype(mx.float64)
        g = mx.grad(f)(x).reshape(-1)
        E = mx.eye(x.size, dtype=mx.float64)
        fd = mx.stack([(f(x + eps * E[i].reshape(x.shape)) - f(x - eps * E[i].reshape(x.shape))) / (2 * eps)
                       for i in range(x.size)])
        err = mx.abs(g - fd).max().item()
    assert err < tol, err
```

With `tanh` and `log1p` this reached 1.4e-10. On 0.32.4, a function that goes through `exp`, `sin`, `cos` or `erf` fails it with errors from 3e-3 to 5e-2, because those ops compute float64 at float32 precision and the `eps=1e-6` difference drowns in their rounding. Use torch as the reference there, or check on a later release.

Finite differences in float32 have a noise floor: the best step measured was `eps=1e-3`, with error ~1e-4; `eps=1e-4` was worse (1.6e-3). Never set a float32 finite-difference tolerance below 1e-3.

Gradients of quantized or integer paths do not exist; test that a frozen quantized base receives no gradient rather than comparing one.

## Benchmarks

A number with no method attached is not evidence.

```python
import time, statistics, mlx.core as mx

def bench(fn, *args, warmup=10, iters=100):
    for _ in range(warmup):
        mx.eval(fn(*args))
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(fn(*args))
        times.append(time.perf_counter() - t)
    return 1e3 * statistics.median(times), 1e3 * min(times)
```

- **Evaluate inside the timed region.** Building `a @ b` for 4096x4096 took 0.03 ms; evaluating it took 11 ms. `mx.async_eval` returns after scheduling (0.17 ms) and measures nothing; follow it with `mx.synchronize()` before stopping the clock.
- **Evaluate the inputs before the clock starts**, or the benchmark includes generating them.
- **Warm up.** The first evaluation of each kernel and shape builds a Metal pipeline (26 ms vs 11 ms for the same matmul here), and the first call of an `mx.compile`d function traces (`mlx-transforms`). Report first-call latency separately when it matters, for example time to first token.
- **Report the median and the spread.** Three back-to-back runs of the same 4096² matmul in one process gave medians of 11.4, 12.0 and 18.4 ms. Interleave A and B in the same process, repeat the pair, and compare medians; check that nothing else is using the GPU.
- Derive throughput from the median: `2*M*N*K / t` for matmul, bytes moved over time for bandwidth-bound ops, which is most decode work.
- For LLM generation use `mlx_lm.benchmark --model M -p 2048 -g 128 -n 5`, which reports prompt and generation throughput separately.

## Model ports

Three checks, in this order. All run without a download by building a tiny random checkpoint with transformers:

```python
import numpy as np, torch, mlx.core as mx
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache

path = Path("tiny-llama")                  # load_model needs a Path, not a str
torch.manual_seed(0)
cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
LlamaForCausalLM(cfg).eval().save_pretrained(path)
hf = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
model, _ = load_model(path)
ids = [[1, 5, 17, 3, 42, 8]]

# 1. Logits on identical weights, in float32.
with torch.no_grad():
    ref = hf(torch.tensor(ids)).logits.numpy()
assert np.abs(np.array(model(mx.array(ids))) - ref).max() < 1e-4

# 2. Greedy continuation, token for token, through the cache.
want = hf.generate(torch.tensor(ids), max_new_tokens=20, do_sample=False)[0, len(ids[0]):].tolist()
cache, y, got = make_prompt_cache(model), mx.array(ids), []
for _ in range(20):
    y = mx.argmax(model(y, cache=cache)[:, -1], axis=-1, keepdims=True)
    got.append(y.item())
assert got == want, next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)

# 3. Cache consistency: one full forward equals prefill plus single-token steps.
full = model(mx.array(ids))
cache = make_prompt_cache(model)
steps = [model(mx.array(ids)[:, :3], cache=cache)] + \
        [model(mx.array(ids)[:, i:i + 1], cache=cache) for i in range(3, len(ids[0]))]
assert mx.allclose(full, mx.concatenate(steps, axis=1), atol=1e-4)
```

Here the logits matched to 1.9e-7 and all 20 greedy tokens matched. Report the **index of the first divergent token**, not a percentage. On real weights in bf16 an early divergence after a near-tie is expected; a divergence at token 0, or a cache-consistency failure, is a bug (RoPE offset, mask, or sanitize).

Then run the shape of mlx-lm's `model_test_runner` (`tests/test_models.py`) on the new architecture: for float32 and float16 the output dtype must equal the weight dtype with and without a cache, a single-token step after the prompt must work, batch 2 must work, and `copy.deepcopy(model)` must succeed. A new architecture PR upstream also needs a reference-output comparison against transformers.

## How MLX tests itself

- Python tests live in `python/tests/test_*.py` in the mlx repo. `mlx_tests.MLXTestCase` adds `assertEqualArray` (shape, dtype, then `allclose`, default `atol=rtol=1e-2`) and `assertCmpNumpy`. It is a repo file, not part of the installed package: copy the pattern, do not import it.
- `DEVICE=cpu python python/tests/run.py` (`pytest` on main) reruns the suite on the CPU backend through `MLXTestCase.setUp`.
- `run.py` (`conftest.py` on main) turns TF32 off for the whole run, and the runner calls `mx.clear_streams()` at the end.
- Distributed tests run under the launcher: `mlx.launch -n 8 python python/tests/ring_test_distributed.py`, and CI fails on any `[WARN]` in stderr.

## CI

- **GPU tests need Apple-silicon Macs with Metal access.** ml-explore runs mlx and mlx-lm on self-hosted Macs. Put `assert mx.metal.is_available()` first in a GPU job; otherwise a runner without a Metal device runs everything on the CPU and passes.
- Run GPU pytest serially; MLX's CI uses `pytest -n 4` only for CPU or self-hosted runs.
- Both repos run tests with `METAL_DEVICE_WRAPPER_TYPE=1` and `METAL_DEBUG_ERROR_MODE=0`; copy those into the job environment so Metal API misuse surfaces in CI.
- A fast lane on Linux: `pip install mlx[cpu]` (glibc 2.35+) runs graph, transform and numerical tests on ordinary runners. It never executes a Metal kernel, so it cannot stand in for the Mac job.
- Matrix the precision switches, not just the code: at least one job with `MLX_ENABLE_TF32=0` comparing against references, and the shipped defaults for the end-to-end tests.
- Distributed code gets a two-rank job on one Mac (`mlx.launch -n 2 --python "$(which python)" tests/test_dist.py`), as mlx-lm does for model-parallel tests (`mlx-distributed`).
