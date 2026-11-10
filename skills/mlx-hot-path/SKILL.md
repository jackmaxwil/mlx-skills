---
name: mlx-hot-path
description: MLX GPU performance — find the slow kernel before changing code, Metal capture and xctrace Metal System Trace with headless export, MLX_METAL_DEBUG builds, which shapes take the fast matmul, gemv, quantized qmv/qmm and fused SDPA paths and which fall back, force_fused errors, command-buffer knobs and env vars read once, launch overhead and compile fusion, Apple GPU hardware rates by generation, when a custom kernel is not worth it. Load before optimizing MLX code on Apple silicon or when a model is slower than expected.
---

# MLX GPU hot path

Written against **mlx 0.32.4** on macOS 26.2+, with numbers measured on an M4 Max (`applegpu_g16s`). Timing method (eval, synchronize, warmup, the JIT first call) belongs to `mlx-verify`. The TF32 default belongs to `mlx`. Writing kernels belongs to `mlx-kernel`. Memory belongs to `mlx-memory`.

## Order of work

| Situation | Do |
| --- | --- |
| "It is slow", with no profile | time each stage with `mx.eval` and `mx.synchronize()` around it, then run one Metal System Trace |
| Many tiny ops (norms, activations, rope, sampling) | `mx.compile` the block first; launch cost dominates, not math |
| One matmul or attention dominates | check its shape against the routing tables below before touching code |
| Decode is slow but prefill is fine | check qmv eligibility (K alignment, batch size) and SDPA vector-kernel eligibility |
| A kernel is slow and on the fast path already | Metal capture: per-kernel GPU time, occupancy, limiter counters |
| Still slow after routing and fusion | then consider a custom kernel (`mlx-kernel`) |

## Metal System Trace (whole run, no rebuild)

```
xcrun xctrace record --template 'Metal System Trace' --output run.trace --launch -- \
    .venv/bin/python bench.py
open run.trace                                     # Instruments GUI
xcrun xctrace export --input run.trace --toc       # list the tables
xcrun xctrace export --input run.trace \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' > gpu.xml
```

Useful schemas in the table of contents:

| Schema | What it holds |
| --- | --- |
| `metal-gpu-intervals` | per-encoder GPU start and end times; gaps between them are CPU-bound launch time |

Use `xcrun xctrace list templates` to check the template names; "Time Profiler" covers the CPU side (Python overhead, graph building).

## Metal capture (a few iterations, full detail)

```python
# Run as: MTL_CAPTURE_ENABLED=1 python bench.py
mx.eval(warm_up())                       # keep JIT compilation out of the capture
mx.metal.start_capture("step.gputrace")  # path must end in .gputrace and must not exist yet
mx.eval(step())
mx.metal.stop_capture()
```

| Error | Cause |
| --- | --- |
| `[metal::start_capture] Failed to start: Capture layer is not inserted.` | `MTL_CAPTURE_ENABLED=1` missing from the environment at process start |
| `... couldn’t be saved in the folder “...” because a file with the same name already exists.` | delete the old `.gputrace` first |
| `‘t3’ could not be used because the file name is invalid.` | the path has no `.gputrace` extension |

Captures are large: three small matmuls produced a 210 MB trace. Capture one step, not a training loop. Open it in Xcode; the Dependencies view shows each kernel with its buffers.

With a release wheel, kernels appear under their MLX names (for example `steel_gemm_...`, `sdpa_vector_...`) but without source. A source build with `CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" pip install -e .` records shader source for the debugger and labels command queues and other Metal objects (source builds: `mlx-native`).

## Matmul routing

| Shape (after broadcasting) | Kernel | Note |
| --- | --- | --- |
| M = N = 1, contiguous | dot product | |
| min(M, N) = 1 | gemv | memory-bound: time is bytes of the matrix over bandwidth |
| 2 to 15 vectors with transposed B, fp16/bf16 | `gemv_wide` | M3 and later only |
| batch 1, K ≥ max(M, N), K/16 ≥ 8, few output tiles | split-K GEMM | non-NAX only; up to 2048 16×16 output tiles on `s`/`d` parts, 1024 elsewhere |
| everything else | steel GEMM | tile sizes chosen per architecture suffix and dtype |
| M5 class (gen ≥ 17, ≥ 18 for `p` (phone) parts), macOS 26.2+, NAX-enabled build | NAX GEMM | Metal 4 tensor ops |

- On NAX (M5) and CUDA, float32 matmul uses TF32 by default (`see mlx`).
- Transposed operands are free: steel GEMM and gemv have kernels for both layouts, so do not materialize `w.T`.

## Quantized matmul routing (`mx.quantized_matmul`, `nn.QuantizedLinear`)

| Condition (transposed weights, the default) | Kernel |
| --- | --- |
| rows of x (M) ≥ batch limit | `qmm` (tiled GEMM); split-K for small M on a non-batched call |
| M < batch limit and K is 64 or 128 | `qmv_quad` |
| M < batch limit, M ≥ 2 | `qmv_wide` (affine mode on M1/M2: `qmv`) |
| M = 1 and K divisible by `pack_factor * 2 * 32` (512 for 4-bit, 2-bit uses 1 instead of 2) | `qmv_fast` |
| M = 1 otherwise | `qmv` (generic, slower inner loop) |
| non-transposed weights with K < 1024 | `qvm` |

- The batch limit depends on generation, the `d` suffix, and K and N. On `applegpu_g16s` it is 13–15 rows; on Ultra (`d`) parts it is 12–32; on M5 class 13–33. Decode with a speculative draft of 4–8 tokens stays on the vector path.
- `x`, `w`, `scales` and `biases` that are not contiguous are copied before every call. Keep quantized weights contiguous; do not transpose or slice them per step.
- Quantization formats, modes and accuracy belong to `mlx-quant`.

## SDPA routing (`mx.fast.scaled_dot_product_attention`)

| Query length qL | Fused kernel | Head dims | Extra conditions |
| --- | --- | --- | --- |
| qL ≤ 8 | vector | 64, 96, 128, 192, 256, 512, and q/v pairs 192/128, 96/64 | qL ≤ kL and qL × (heads / kv_heads) ≤ 32. D = 512 by default: only qL = 1, GQA factor 8, kL ≥ 1024, no array mask |
| qL > 8 | full (tiled) | 64, 72, 80, 96, 128, 192, 256 (512 with NAX), and 96/64 | matching q and v dims except 96/64 |

Falls back to the unfused path (matmul, softmax, matmul, so the full score matrix is materialized):

- qL > 8 with D = 192 or 256, except causal D = 256 with qL ≥ 2048 and qL = kL in fp16 or bf16 on non-NAX hardware.
- qL ≤ 8 with D = 192 and equal q and v dims.
- Any head dim not in the table (100, 112, 160, …). At D = 100 a decode step took 0.53 ms against 0.22 ms at D = 128 on the same shape.

`force_fused=True` (since 0.32.2) raises instead of falling back. Use it in a test to prove the fast path is taken:

```
the vector attention kernel requires the query length times the GQA factor to be at most 32; got query length ...
the full attention kernel supports head dims {64, 72, 80, 96, 128, 192, 256} with matching query/value dims, o...
the vector attention kernel supports h...
```

The first example is qL = 8 with 32 query heads over 4 KV heads (8 × 8 = 64 > 32). Chunk speculative verification so that qL × GQA ≤ 32, or accept the full kernel for qL > 8.

`MLX_SDPA_BLOCKS=N` overrides the number of key blocks in the two-pass vector kernel (rounded up to a multiple of 32). `MLX_SDPA_D512_MIN_KL` sets the minimum key length for D = 512 (default 1024); `=0` lifts all the D = 512 restrictions.

## Launch overhead and command buffers

Measured: 2000 dependent elementwise ops on a 1024-float array took 12–19 ms (6–9 µs per op). The same function under `mx.compile` took 2–5 ms. For small tensors the GPU is idle most of the time; fusion and fewer, larger ops are the lever, not a faster kernel.

MLX commits a command buffer after N ops or M MB of inputs, whichever comes first:

| Arch suffix | `MLX_MAX_OPS_PER_BUFFER` default | `MLX_MAX_MB_PER_BUFFER` default |
| --- | --- | --- |
| `p` (iPhone, iPad) | 20 | 40 |
| `g` (base M chips) | 40 | 40 |
| `s` (Pro, Max) | 50 | 50 |
| `d` (Ultra) | 50 | 50 |

Measured effect of the ops knob on the 2000-op loop: `=1000` 13–15 ms (no gain), `=10` 34 ms, `=1` 74 ms, `=0` 90–129 ms. **An empty value counts as 0**: `MLX_MAX_OPS_PER_BUFFER=` in a shell or a `.env` file commits after every op and was about 7× slower. Leave both unset unless a trace shows commit gaps.

Most `MLX_*` environment variables are read once, on first use, and cached for the process. Setting `os.environ[...]` after the first eval has no effect; set it before `import mlx.core` or in the launching shell. The read-once list includes `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`, `MLX_ENABLE_TF32`, `MLX_METAL_GPU_ARCH`, `MLX_METAL_FAST_SYNCH`, and the residency knobs in `mlx-memory`. `MLX_SDPA_BLOCKS`, `MLX_SDPA_D512_MIN_KL` and `MLX_CONV_WINOGRAD` are read on every call. `mx.config` (0.32.5, not in the 0.32.4 wheel) reads and changes settings at runtime.

## Apple GPU rates that change design

| Fact | Consequence |
| --- | --- |
| SIMD width 32 on every generation | threadgroups in multiples of 32; reductions with `simd_sum` |
| at most 1024 threads per threadgroup, 32 KB threadgroup memory | a tile of 64×64 fp32 (16 KB) twice is already the whole budget |
| about 208 KB of registers per core; ALU saturates near 24 simdgroups per core | heavy register use lowers occupancy; on M1/M2 it lowers `maxTotalThreadsPerThreadgroup` below 1024 |
| fp16 and fp32 FMA run at the same rate (M1 and later) | half precision saves bandwidth and registers, not ALU time |
| `simdgroup_matrix` about 102 FFMA per core-cycle for both f16 and f32 | the matrix units gain from layout and reuse, not from dtype |
| bf16 has hardware conversion only on Apple9 (M3) and later | on M1/M2, bf16 is decoded in software; prefer fp16 for compute on those chips |
| int32 multiply runs at 1/4 the add rate; int64 add 1/4, int64 multiply 1/16 | keep index math in 32-bit; precompute strides; MLX switches to 64-bit indexing only past `INT32_MAX` |
| exp2 and log2 about 4 cycles, reciprocal 6, rsqrt 8, sin about 14 | softmax in base 2 with a prescaled `log2(e)` is cheaper than `exp` |
| float atomics are emulated and slow | scatter-add backward passes serialize; prefer gather or two-pass designs |
| 128-byte cache line | coalesce: adjacent lanes read adjacent addresses |
| `applegpu_g13`…`g17` = M1…M5; suffix `p` phone, `g` base, `s` Pro/Max, `d` Ultra | gate tuning on `mx.device_info()["architecture"]`, not on the marketing name |

The register, simdgroup, FFMA and cycle figures are M1–M2 measurements. On M3 and later, dynamic caching allocates registers and threadgroup memory at runtime, so occupancy no longer follows these static budgets; check the pipeline's `maxTotalThreadsPerThreadgroup` and profile.

Bandwidth, not FLOPs, bounds decode: a 4-bit 8B model streams about 4.5 GB per token, so tokens per second is close to memory bandwidth divided by model bytes. Compare against that before chasing kernel time.

- Prefill is compute-bound, which is where M5 Neural Accelerators help: Apple reports 3.3–4.1× faster time-to-first-token than M4. Decode is bandwidth-bound and gains only 19–27%, tracking 120→153 GB/s, so for decode optimize bytes moved, not FLOPs.

## When not to write a kernel

| Temptation | Better |
| --- | --- |
| Fusing an elementwise chain | `mx.compile` fuses elementwise ops already |
| Custom attention for an unusual head dim | pad the head dim to the next supported size (100 to 128); padding costs less than the unfused fallback |
| Custom decode matmul | check `qmv_fast` alignment first: pad K to a multiple of 512 for 4-bit |
| Custom RMSNorm, LayerNorm, RoPE, softmax | `mx.fast.rms_norm`, `layer_norm`, `rope`, `mx.softmax(precise=...)` exist and are tuned |
| Faster small ops at batch 1 | batch more requests; launch overhead does not shrink with a better kernel |
| "Metal 4 tensor ops will make it fast" on M1–M4 | NAX exists only on M5 class with macOS 26.2; elsewhere the 8×8 `simdgroup_matrix` path is the ceiling |

## Anti-patterns

| Anti-pattern | Fix |
| --- | --- |
| Optimizing before a Metal System Trace | trace first; launch gaps and fallbacks are the usual cause |
| Timing the first call | warm up; the first call includes JIT compilation (`mlx-verify`) |
| A capture of a whole training loop | capture one step |
| `MLX_MAX_OPS_PER_BUFFER=` left empty in a `.env` | unset it |
| Setting `MLX_*` variables after the first eval | set them before `import mlx.core` |
| Head dims outside the fused SDPA table | pad, or check with `force_fused=True` |
| Non-contiguous quantized weights | keep them contiguous; the copy runs every call |
| bf16 compute on M1/M2 | fp16 |
| Tuning by device name | tune by architecture string and suffix |
