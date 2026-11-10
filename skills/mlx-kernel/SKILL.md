---
name: mlx-kernel
description: Custom Metal kernels for MLX — mx.fast.metal_kernel signature generation, grid vs threadgroup, template vs runtime arguments, small and strided inputs, uninitialized outputs, init_value and atomics, math_mode, vjp through custom_function, NAX/tensor_ops gating, shader validation and printf, C++/Metal extension primitives and the encoder contract. Load when writing or debugging mx.fast.metal_kernel or a C++/Metal extension.
---

# Custom Metal kernels

Written against **mlx 0.32.4** on macOS 26.2+. Before writing a kernel, check that MLX does not already have a fast path for the shape, and profile first (`mlx-hot-path`). Correctness tests and benchmark method are in `mlx-verify`. This skill covers making a kernel correct and keeping it cheap to launch.

## Build the kernel once

`mx.fast.metal_kernel(...)` returns a callable. Each distinct `(template values, input dtypes, input kinds, output dtypes)` combination compiles a new Metal library the first time it runs. On an M4 Max that took **85–100 ms**; later calls took 0.2–6 ms.

```python
# Wrong: rebuilds the callable, and the source string, on every forward pass.
def rms(x):
    k = mx.fast.metal_kernel(name="rms", input_names=["x"], output_names=["out"], source=SRC)
    return k(...)[0]

# Right: module-level, or lru_cache keyed on the variant (mlx-lm's sparse_mla.py does this).
_rms = mx.fast.metal_kernel(name="rms", input_names=["x"], output_names=["out"], source=SRC)

@lru_cache(maxsize=None)
def _kernel(variant):
    return mx.fast.metal_kernel(name=f"smla_{variant.name}", ..., header=_HEADER + variant.header)
```

## What the generated signature looks like

You write only the body. MLX generates the signature from the call. Use `verbose=True` once to print it.

| You pass | The kernel sees |
| --- | --- |
| array input with **8 or more** elements | `const device T* name` |
| array input with **fewer than 8** elements | `const constant T* name` |
| 0-d array, or a Python scalar | `const constant T& name` (Python `float` → `float`, `int` → `int32_t`, `bool` → `bool`) |
| `name_shape` appears anywhere in the source | `const constant int* name_shape` |
| `name_strides` appears anywhere in the source | `const constant int64_t* name_strides` (element strides) |
| `name_ndim` appears anywhere in the source | `const constant int& name_ndim` |
| output | `device T* name`, or `device atomic<T>* name` for **every** output when `atomic_outputs=True` |
| `thread_position_in_grid`, `simdgroup_index_in_threadgroup`, … appears in the source | that attribute as a parameter (`uint3` for the position and size attributes, `uint` for the index attributes) |
| `template=[("T", mx.float16), ("D", 128), ("B", True)]` | `template <typename T, int D, bool B>` plus an explicit instantiation |

Dtype names inside the kernel: `float`, `float16_t`, `bfloat16_t` (which is the native `bfloat`), `int32_t`, `uint32_t`, `int64_t`, and so on. `float64` inputs fail with `ValueError: float64 is not supported on the GPU`.

Detection is plain substring search over the source, including comments. The resulting errors look like this:

- An input `x` plus a local `int x_ndim = 3;` fails with `error: redefinition of 'x_ndim' with a different type: 'thread int' vs 'const constant int &thread'`.
- Reusing a generated name for your own variable fails the same way.
- Naming a template parameter after an input fails the same way. mlx-lm's gated_delta kernel names its type parameter `InT` because `T` is an input.

The preamble is MLX's `kernels/utils.h`: bf16, complex, `elem_to_loc`, `ceildiv`, and the 64-bit `simd_shuffle` overloads. `#include "mlx/..."` does not resolve, because the source is compiled at runtime with no include path. System headers do resolve: `<metal_stdlib>`, `<metal_simdgroup_matrix>`, `<metal_atomic>`, `<metal_logging>`, `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`. Put helpers in `header=`.

Compile errors are reported against `mlx/backend/metal/kernels/utils.h:<line>`, because your code is appended to that file. The quoted source line in the message is the one to read, not the file name.

## Small inputs live in `constant` memory

A helper typed for `device` pointers compiles for every test until one passes a 4-element array:

```metal
// Wrong: fails for inputs with fewer than 8 elements:
//   error: no matching function for call to 'first'
//   note: candidate template ignored: deduced type 'const device float *thread' of 1st
//   parameter does not match adjusted type 'const constant float *' of argument
template <typename T> T first(const device T* p) { return p[0]; }

// Right: generic over the pointer type.
template <typename P> auto first(P p) { return p[0]; }
```

Pass per-call scalars such as `eps`, a scale or a length as 0-d inputs. mlx-lm caches them with `@lru_cache def scalar(v, dtype=mx.float32): return mx.array(v, dtype=dtype)`.

## Template or runtime input

| Value | Pass as | Why |
| --- | --- | --- |
| Head dim, group size, rows per threadgroup: fixed per model | `template` int | compile-time loop bounds, unrolling, and `/` `%` turned into shifts |
| Sequence length, token count, batch: changes per call | 0-d input (`inputs=[..., T]`) | a template value that changes per call recompiles every call (about 90 ms each) |
| Element type | `template` `("T", x.dtype)` | one source for f16, bf16 and f32 |
| Feature switches (mask, causal) | `template` bool, or separate kernels | the dead branch is compiled out |

## Grid is total threads

`grid` and `threadgroup` go to `dispatchThreads`. That means `grid` is the **total number of threads**, not the number of threadgroups. This is the opposite of MLX's own C++ kernels, which mostly call `dispatch_threadgroups`.

```python
# Intent: one threadgroup of 256 per row of R rows.
# Wrong: launches R threads in total, so only row 0 gets a threadgroup. Measured: rows 1-7 never written.
k(inputs=[x], grid=(R, 1, 1), threadgroup=(256, 1, 1), ...)
# Right: R threadgroups of 256 threads. Read the row from threadgroup_position_in_grid.x.
k(inputs=[x], grid=(R * 256, 1, 1), threadgroup=(256, 1, 1), ...)
```

| Pattern | grid | threadgroup | Index |
| --- | --- | --- | --- |
| One thread per element | `(n, 1, 1)` | `(256, 1, 1)` | `thread_position_in_grid.x` |
| One simdgroup per row (D ≤ about 1024) | `(rows * 32, 1, 1)` | `(256, 1, 1)` | `row = thread_position_in_grid.x / 32`, `lane = thread_index_in_simdgroup` |
| One threadgroup per row | `(rows * TG, 1, 1)` | `(TG, 1, 1)` | `threadgroup_position_in_grid.x`, `thread_position_in_threadgroup.x` |
| 2-D tiles (mlx-lm gated_delta) | `(32, Dv, B * Hv)` | `(32, 4, 1)` | `.x` lane, `.y` column, `.z` batch×head |

- A grid that is not a multiple of the threadgroup is fine. Metal launches partial threadgroups at the edge and never runs a thread outside the grid; `grid=(1000,)` with `threadgroup=(256,)` wrote exactly 1000 elements. `simd_sum` in a partial simdgroup sums only the live lanes. A bounds guard is still needed when you derive indices from `threadgroup_position_in_grid * TG` instead of the grid position.
- Each grid dimension is a C `int`. `grid=(2**32, 1, 1)` raises `TypeError: __call__(): incompatible function arguments`. Spread more than 2^31 threads over y and z.
- The threadgroup is clamped to the grid per axis. A threadgroup above the pipeline's maximum raises `ValueError: Thread group size (2048) is greater than  the maximum allowed threads per threadgroup (1024).` (the double space is in the message).
- The maximum can be below 1024 for a register-heavy kernel on M1/M2. MLX ships TK=1 SDPA variants for exactly that case. You cannot add `max_total_threads_per_threadgroup` to a generated signature, so use 256 or 512 for heavy kernels, and never size a threadgroup from the data: mlx-lm's plamo3 uses `threadgroup=(hidden_size // 4, 1, 1)`, which breaks once hidden_size goes past 4096.
- The SIMD width is 32 on every Apple GPU. Keep threadgroups a multiple of 32, and reduce with `simd_sum` and then threadgroup memory: mlx-lm `kernels/_common.py` `block_sum`. Threadgroup memory is declared statically in the body (`threadgroup float buf[32];`). `metal_kernel` has no dynamic shared-memory argument; `cuda_kernel` does.

## Strided inputs

By default, `ensure_row_contiguous=True` copies every input whose `row_contiguous` flag is false (transposed, broadcast, or step-sliced views) with an extra copy kernel. A row-contiguous view with an offset, such as `x[1:]`, is bound with its offset and not copied. To skip the copy, index through the strides:

```metal
// Wrong for transposed, broadcast and step views: reads the wrong elements.
out[i] = inp[i];
// Wrong for negative strides (x[:, ::-1]): elem_to_loc deduces IdxT from the uint argument.
uint loc = elem_to_loc(i, inp_shape, inp_strides, inp_ndim);   // also the upstream doc example
// Right for all four, verified against transpose, broadcast, step and reversed views.
int64_t loc = elem_to_loc<int64_t>(i, inp_shape, inp_strides, inp_ndim);
out[i] = inp[loc];
```

Outputs are always fresh, row-contiguous buffers.

## Outputs start as garbage

Outputs are allocated, not zeroed. A kernel that writes half the elements returns stale cache contents in the other half; the measured result was `7.0`, left over from a freed array. Either write every element, or pass `init_value=0`, which costs one fill kernel per output.

## Atomics

`atomic_outputs=True` turns **all** outputs into `device atomic<T>*`. Metal atomics exist for `int`, `uint` and `float` only. With a `float16` or `bfloat16` output the kernel fails to compile:

```
error: no matching function for call to 'atomic_fetch_add_explicit'
note: candidate template ignored: requirement '_valid_fetch_add_sub_type<thread half *, void>::value' was not satisfied
```

Accumulate in a `float32` output and cast afterwards. Reduce inside the simdgroup first, so there is one atomic per 32 lanes:

```metal
float v = simd_sum(float(inp[thread_position_in_grid.x]));
if (thread_index_in_simdgroup == 0)
    atomic_fetch_add_explicit(&out[0], v, memory_order_relaxed);
```

Always combine this with `init_value=0`. Apple GPUs have no fast float atomics (metal-flash-attention calls `atomic<float>` emulated), so a backward pass built on scatter-add atomics is the slow design. Prefer a two-pass or recompute scheme.

## math_mode (since 0.32.0)

`compile_options={"math_mode": "safe" | "relaxed" | "fast"}`, with `safe` as the default. `relaxed` needs macOS 15. On an M4 Max, `metal::exp(-inf)` returned 0 in all three modes. What `fast` changed was NaN and Inf reasoning: `isnan(x - x)` for `x = -inf` returned 1 under safe and relaxed, and 0 under fast, because the compiler folds `x - x` to 0. Keep `safe` for any kernel that sees masked `-inf` scores or that propagates NaN.

## Gradients and transforms

| Transform | Result on a bare kernel call |
| --- | --- |
| `mx.grad` | `ValueError: [Primitive::vjp] Not implemented for CustomKernel.` |
| `mx.vmap` | `ValueError: [Primitive::vmap] Not implemented for CustomKernel.` |
| `mx.compile` | works, and the kernel is a fusion barrier |
| `mx.compile(shapeless=True)` | `ValueError: [Primitive::output_shapes] CustomKernel cannot infer output shapes.` on the **first** call |

Wrap the kernel in `mx.custom_function` and attach the backward pass. With one positional input, `primals` is the array itself, not a tuple, and you return a single array; `x, = primals` raises `ValueError: too many values to unpack`.

```python
@mx.custom_function
def sq(x):
    return _sq(inputs=[x], grid=(x.size, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

@sq.vjp
def sq_vjp(primals, cotangent, output):
    x = primals                      # a tuple only when the function takes two or more inputs
    return 2 * x * cotangent
```

The rules for `custom_function`, and how it composes with `compile`, belong to `mlx-transforms`.

## Per-generation variants

An unspecified stream runs a custom kernel on the GPU, even inside `with mx.stream(mx.cpu)`. Gate by architecture string, never by device name:

```python
arch = mx.device_info()["architecture"]           # "applegpu_g16s" = M4 Max
gen = int(re.search(r"applegpu_g(\d+)", arch).group(1))
# NAX / Metal 4 tensor ops: gen >= 17 (M5), or >= 18 for "p" (phone) parts, and macOS >= 26.2.
use_tensor_ops = gen >= (18 if arch.endswith("p") else 17) and platform.mac_ver()[0] >= "26.2"
```

That string comparison is lexicographic; parse the version into a tuple, as mlx-lm's `sparse_mla.is_available` does. Ship two fragment implementations behind one body:

- **`tensor_ops`**: `mpp::tensor_ops::matmul2d<desc(16, 32, 16, ...), metal::execution_simdgroup>` with cooperative tensors, the same shape MLX's NAX kernels use (`kernels/steel/gemm/nax.h`).
- **8×8 `simdgroup_matrix`**: runs on every Apple GPU since M1.

Accumulate in float32 in both. Use the GPU-side hardware rules in `mlx-hot-path` to choose the tile size.

## Debugging a kernel

- `verbose=True` prints the generated source on every call.
- Out-of-bounds writes are silent: a store at byte 9216 of a 4096-byte output returned normally and reported nothing. Run once under shader validation:

  ```
  MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 python repro.py
  ```

  Without `..._REPORT_TO_STDERR` nothing is printed, and even with it the program continues and returns. The report names the kernel and the generated line:

  ```
  Invalid Metal usage: Invalid store in 'device' or 'constant' address space at offset 9216, executing compute function: "custom_kernel_oob_float_float"
  MTLBuffer: <no label>, length:4096, ...
  GPU stack trace:
      * frame #0: custom_kernel_oob_float_float() - /mlx/backend/metal/kernels/utils.h:543:51
  ```

- `MTL_DEBUG_LAYER=1` enables Metal API validation ("Metal API Validation Enabled"); ordinary MLX workloads run clean under it.
- Shader printf is a no-op in the PyPI wheel. Both `metal::os_log_default.log(...)` and `mlx::os_log` compile and print nothing, because MLX enables Metal logging only when MLX itself is a debug build (`DEBUG=1 pip install -e .`). Then run with `MTL_LOG_LEVEL=MTLLogLevelDebug MTL_LOG_TO_STDERR=1`. In a release build, debug by writing to an extra output array.

## C++/Metal extension primitives

Use an extension when the kernel needs function constants, a precompiled metallib, CPU and GPU implementations, or its own vjp, jvp and vmap on the primitive. The skeleton is `mlx/examples/metal_extension`, which uses `mlx.extension.MetalExtension` (since 0.32.4). The CMake form is `examples/cmake_extension` with `mlx_build_metallib`.

```python
# setup.py; build with: python setup.py build_ext -j8 --inplace
setup(ext_modules=[extension.MetalExtension("my_ext", sources=["op.cpp", "op.metal"],
                                            extra_compile_args={"cxx": ["-O3"], "metal": ["-O3"]})],
      cmdclass={"build_ext": extension.BuildExtension})
```

Build requirements are `setuptools`, `cmake>=3.25`, `nanobind==3.1.0` and an `mlx` matching the runtime. The build produces `my_ext.cpython-*.so`, `my_ext.metallib` and `my_ext.pyi` side by side. `.metal` sources get MLX's include directories, so `#include "mlx/backend/metal/kernels/utils.h"` works there, unlike in `metal_kernel`. The Metal Toolchain and source builds are covered in `mlx-native`.

```cpp
void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& d = mx::metal::device(stream().device);
  auto lib = d.get_library(MLX_METAL_LIBRARY_NAME, current_binary_dir());  // dladdr dir
  auto kernel = d.get_kernel("my_kernel_f16", lib);
  auto& enc = mx::metal::get_command_encoder(stream());
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(inputs[0], 0);
  enc.set_output_array(out, 1);
  enc.set_bytes(n, 2);
  size_t tg = std::min<size_t>(n_threads, kernel->maxTotalThreadsPerThreadgroup());
  enc.dispatch_threads(MTL::Size(n_threads, rows, 1), MTL::Size(tg, 1, 1));
}
```

MLX allocates buffers as hazard-untracked and inserts its own barriers and fences, based on which `MTL::Buffer` each dispatch reads and writes. The encoder calls are that bookkeeping:

| Call | Contract |
| --- | --- |
| `set_input_array(a, idx)` | binds `a` with its offset and records a read; a barrier is inserted if an earlier dispatch wrote this buffer |
| `set_output_array(a, idx)` | records a write. Raw `setBuffer` or `set_buffer` on an MLX array skips this, and the next kernel can race |
| `add_temporary(arr)` / `add_temporaries(v)` | keeps scratch arrays alive until the command buffer completes; without it they are freed while the GPU still reads them |
| `set_bytes(v, idx)` / `set_vector_bytes(vec, idx)` | small constants; an empty vector is bound as one default value |
| `dispatch_threads(grid, group)` | grid in threads |
| `dispatch_threadgroups(grid, group)` | grid in threadgroups (steel GEMM, SDPA) |
| `get_kernel(base, lib, hash_name, func_consts)` | the pipeline is cached per `(library, hash_name)`: encode every function constant into `hash_name`, or a second specialization returns the first one's pipeline |

- Never create your own `MTLCommandQueue` or `MTL4CommandQueue` for MLX arrays; MLX's fences cannot see that work.
- Gate NAX code with `mx::metal::is_nax_available()`. NAX kernels are compiled in only with SDK and `MACOSX_DEPLOYMENT_TARGET` ≥ 26.2.

Load-time errors and their meaning:

| Message | Cause |
| --- | --- |
| `Failed to load the metallib from <.../my_ext.metallib> with error library not found` | the `.metallib` is not next to the `.so` |
| `[metal::Device] Unable to load kernel my_kernel_bf16` | the name matches no `[[host_name]]` or `kernel` function in the metallib |
| `[metal::Device] Unable to build metal library from source` | a JIT source failed to compile (`metal_kernel`, or a `get_library(name, builder)` call) |

## Anti-patterns

| Anti-pattern | Fix |
| --- | --- |
| `metal_kernel(...)` inside the forward function | build once at module level, or with `lru_cache` |
| `template=[("L", seq_len)]` for a value that varies per call | pass it as a 0-d input |
| `grid=(n_threadgroups, 1, 1)` | `grid=(n_threadgroups * tg, 1, 1)` |
| `threadgroup=(dim // 4, 1, 1)` | a fixed 256 or 512 with a loop over `dim` |
| `const device T*` helper parameters | template the helper on the pointer type |
| `uint loc = elem_to_loc(...)` | `int64_t loc = elem_to_loc<int64_t>(...)` |
| Partial writes with no `init_value` | write every element, or `init_value=0` |
| `atomic_outputs=True` with an f16 or bf16 output | float32 output, `simd_sum` first, cast after |
| `mx.grad` through a bare kernel | `mx.custom_function` with `.vjp` |
| `compile_options={"math_mode": "fast"}` on masked softmax | keep `safe` |
| Raw `setBuffer` for an MLX array in an extension | `set_input_array` / `set_output_array` |
| A scratch array that is not registered | `add_temporary` |
