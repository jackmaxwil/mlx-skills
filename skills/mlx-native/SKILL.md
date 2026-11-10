---
name: mlx-native
description: MLX outside Python and Swift — choosing between the wheel's C++ library, a source build and mlx-c, C++ embedding with CMake through `python -m mlx --cmake-dir`, the metallib search order and override, source builds, deployment targets and NAX gating, the Metal Toolchain, mlx-c handle ownership, out-params, data access, the error handler, closures and payloads, thread-affine streams in bindings, Rust/Go/Zig binding rules, CUDA and Linux deltas. Load when linking MLX into C, C++, Rust, Go or Zig, writing or reviewing a binding, building MLX from source, or debugging "Failed to load the default metallib".
---

# MLX from native code

Written against **mlx 0.32.4** and **mlx-c 0.8.0**.

## Pick the path

| You need | Use | Needs the Metal Toolchain |
| --- | --- | --- |
| C++ program on a Mac that has the wheel | `find_package(MLX)` against `python -m mlx --cmake-dir` | no: the wheel ships `libmlx.dylib` and `mlx.metallib` |
| C ABI for Rust/Go/Zig | mlx-c built with `-DMLX_C_USE_SYSTEM_MLX=ON` against the same wheel | no |
| Static link, custom flags, CUDA, Linux, a deployment target the wheel does not have | MLX source build (CMake) | yes on macOS |
| Swift app | mlx-swift through SwiftPM or Xcode (`see mlx-swift-app`) | yes |

Pin versions together. mlx-c 0.8.0 is written against mlx 0.32.4, and its own fetch pins `GIT_TAG v0.32.4`. A wheel of a different minor version under `MLX_C_USE_SYSTEM_MLX` is an ABI mismatch, so do not leave it to chance.

## C++ against the wheel

This file builds and runs as written:

```cmake
cmake_minimum_required(VERSION 3.27)
project(example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)
add_executable(example example.cpp)
target_link_libraries(example PRIVATE mlx)
```

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build build && ./build/example
```

- **Pass `-DPython_EXECUTABLE`.** Without it, `find_package(Python)` can pick a system Python that has no mlx, and `MLX_ROOT` comes back empty.
- **The wheel is a shared build.** `lib/` holds `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib`. CMake writes an `LC_RPATH` that points into the venv's `site-packages/mlx/lib`. Deleting or moving that venv breaks the binary. To ship, copy all three files next to each other and set your own rpath, for example `@executable_path/../Frameworks`. The metallib is found because it sits beside `libmlx.dylib` (see the search order below), not beside your executable.
- **A link warning is harmless.** The exported target carries the build machine's SDK framework path, so the link prints `ld: warning: search path '/Applications/Xcode_26.6.app/.../Frameworks' not found`.
- **Check the deployment target.** The macOS wheel installed on macOS 26.2+ is built with `minos 26.2` (`vtool -show-build libmlx.dylib`). 0.32.4 also ships 14.0 and 15.0 wheels (main adds a 27.0 target), and the 14.0 and 15.0 wheels have no NAX kernels.
- `MLX_CXX_FLAGS` (`-DACCELERATE_NEW_LAPACK -D_METAL_`) and the `metal_cpp` include directory reach you through the imported target. Use `target_link_libraries(... mlx)`, not `MLX_LIBRARIES` plus hand-copied flags.

## Metallib search order

The C++ core resolves the kernel library once, on first GPU use, in this order. "Binary dir" is the directory of the Mach-O that contains the MLX code: `libmlx.dylib` for a shared build, your executable for a static link, and the framework binary for `Cmlx.framework`.

| # | Location | Who relies on it |
| --- | --- | --- |
| 1 | `set_metallib_path(path)` / `mlx_metal_set_metallib_path` / Swift `GPU.metallib` | anyone relocating the file |
| 2 | `<binary dir>/mlx.metallib` | wheel, static C++ builds |
| 3 | `<binary dir>/Resources/mlx.metallib` | |
| 4 | `<binary dir>/../Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` | `swift test` under Swift Build |
| 5 | `mlx-swift_Cmlx.bundle/default.metallib` in `Bundle.main`, then every loaded bundle, then the framework whose identifier is `SWIFTPM_BUNDLE` | Swift apps and CLIs |
| 6 | `<binary dir>/Resources/default.metallib` | dynamic-framework packaging |
| 7 | the compile-time `METAL_PATH` | source builds that define it |

Steps 4 and 5 exist only when compiled with `SWIFTPM_BUNDLE`, which mlx-swift sets and the CMake build does not.

- **Override.** It must run before the first GPU operation, because the library is loaded once. A bad override path throws `Can not load metallib from specified location "<path>": ...` and does **not** fall back to the search.
- **Failure.** If every step fails, you get `Failed to load the default metallib.` followed by one `library not found` per attempted location. That error means a packaging bug. Do not catch it and retry: copy `mlx.metallib` next to the binary that contains MLX, or set the override.
- **Static C++ link.** Put `mlx.metallib` beside the executable, or build with `-DMETAL_PATH="/abs/path/mlx.metallib"`.
- `MLX_METAL_JIT=ON` shrinks the metallib by compiling kernels at runtime on first use. The cold start ranges from a few hundred milliseconds to seconds, and the system kernel cache persists across reboots. A smaller metallib still has to be found.

## Source builds on macOS

```bash
xcodebuild -showComponent MetalToolchain     # "Status: installed" or "uninstalled"
xcodebuild -downloadComponent MetalToolchain # Xcode 26+: the metal compiler is a separate download
xcrun metal --version                        # must print a version, not "missing Metal Toolchain"
xcrun -sdk macosx --show-sdk-version         # the SDK CMake will use; DEVELOPER_DIR picks the Xcode
```

The missing-toolchain error is `cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain`. The older form, `unable to find utility "metal"`, means `xcode-select` points at Command Line Tools: run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer`.

```bash
MACOSX_DEPLOYMENT_TARGET=26.2 cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF
cmake --build build && cmake --install build --prefix /opt/mlx   # then MLX_ROOT=/opt/mlx
MACOSX_DEPLOYMENT_TARGET=26.2 pip install .                         # Python wheel from source
```

- **Set the deployment target explicitly.** If `CMAKE_OSX_DEPLOYMENT_TARGET` is empty, the build targets the **host** macOS version. A library built on macOS 27.3 then refuses to load on 26.x. The 0.32.4 release wheels target 14.0, 15.0 and 26.2; main adds 27.0. Anything below 14.0 is a configure error.
- **NAX kernels** (M5 neural accelerators) compile only when the Metal version is at least 4.0, the macOS SDK is at least 26.2 **and** the deployment target is at least 26.2. Otherwise CMake prints `NAX kernels require Metal 4, macOS SDK >= 26.2, and MACOSX_DEPLOYMENT_TARGET >= 26.2 ... Building without NAX kernels.` and defines `MLX_METAL_NO_NAX`. The build succeeds and runs slower on M5. Treat that warning as a failure in any build meant for M5.
- **JACCL** (`see mlx-distributed`) is built only with SDK 26.2 or later and a deployment target of 26.2 or later.
- Useful options: `BUILD_SHARED_LIBS`, `MLX_METAL_JIT`, `MLX_BUILD_CPU`, `MLX_BUILD_SAFETENSORS`, `MLX_BUILD_GGUF`, `USE_ASAN`, `USE_UBSAN`, `USE_TSAN`, `MLX_METAL_DEBUG`. For the smallest binary use `-DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_SHARED_LIBS=ON -DMLX_METAL_JIT=ON`.
- **A self-built metallib can be wrong while the build is green.** Xcode 26.5 / `metalfe-32023.883` mis-compiled quantized GEMM kernels (`qmm`, `qmm_splitk`), and models produced gibberish (mlx#3586). The workaround upstream gave is `MACOSX_DEPLOYMENT_TARGET=26.2` at build time. After any source build on a new toolchain (`xcrun metal --version`), compare greedy outputs or `quantized_matmul` results against the PyPI wheel before shipping (`see mlx-verify`). The quickest isolation is swapping only `mlx.metallib` between the two installs.
- Build from a native arm64 shell. If `uname -p` prints `x86`, the terminal is running under Rosetta.

## Linux and CUDA deltas

| | macOS | Linux CPU | Linux CUDA |
| --- | --- | --- | --- |
| packages | Xcode + Metal Toolchain | `libblas-dev liblapack-dev liblapacke-dev` | those + CUDA toolkit ≥ 12 + `libcudnn9-dev-cuda-12` |
| CMake | default | `-DMLX_BUILD_METAL=OFF` is implied | `-DMLX_BUILD_CUDA=ON`, arch from `MLX_CUDA_ARCHITECTURES` (detected natively if unset; `all` for every arch) |
| pip | `pip install mlx` | `pip install mlx[cpu]` | `pip install mlx[cuda12]` / `mlx[cuda13]` (driver ≥ 580) |
| metallib | required | none | none |

`mlx_metal_is_available` returns false off Apple platforms, so gate Metal-only calls on it rather than on `#ifdef __APPLE__`. Custom Metal kernels and metallib paths do not exist there; `mx.fast.cuda_kernel` mirrors `metal_kernel`'s call shape.

## mlx-c ownership

Every object (`mlx_array`, `mlx_stream`, `mlx_device`, `mlx_vector_*`, `mlx_map_*`, `mlx_string`, `mlx_closure*`) is a `struct { void* ctx; }` pointing at a heap C++ object. The rules:

1. One `*_free` for each `*_new*`. Freeing an empty array (`mlx_array_new()` never assigned) is a no-op.
2. **Results come back through the first pointer argument**, and the return value is an `int` status (0 = ok). The out-param must be either `mlx_array_new()` (empty) or a live handle you own. A live handle is overwritten in place, and its old value is released.
3. **A struct copy is an alias, not a copy.** `mlx_array b = a;` gives two handles to one C++ object, so freeing both is a double free. A real copy is `mlx_array b = mlx_array_new(); mlx_array_set(&b, a);`, and both must be freed.
4. Getters allocate. `mlx_vector_array_get(&res, vec, i)` gives you a handle to free. Adding an array to a vector keeps the array alive until the vector is freed, so free your own handle as usual.
5. `const mlx_array` parameters are borrowed. The callee never frees them.

```c
// WRONG: uninitialized out-param (writes through a garbage ctx); result never freed
mlx_array r;
mlx_add(&r, a, b, s);

// RIGHT
mlx_array r = mlx_array_new();
if (mlx_add(&r, a, b, s) != 0) { /* handle error; r is still yours to free */ }
mlx_array_free(r);
```

**Replacing an unevaluated array that is one output of a multi-output primitive** (a split, a fresh `quantize`): free it and create a new handle; do not `mlx_array_set` over it. `set` is a C++ copy assignment, and it skips the destructor that breaks sibling reference cycles, so the whole graph leaks. mlx-swift fixed exactly this leak.

## Reading data

```c
mlx_array_eval(x);                                 // first: an unevaluated array has no buffer
if (mlx_array_dtype(x) != MLX_FLOAT32) { /* mlx_astype + eval, or read the right type */ }
mlx_array c = mlx_array_new();
mlx_contiguous(&c, x, false, s); mlx_array_eval(c); // or honour mlx_array_strides(x)
const float* p = mlx_array_data_float32(c);        // valid while c is alive
```

- **Calling `mlx_array_data_*` on an unevaluated array segfaults** inside `mlx_array_data_float32`. The header says it "returns NULL"; it does not.
- **There is no dtype check.** `mlx_array_data_float32` on a float16 array returns a valid-looking pointer to reinterpreted bytes, and no error handler fires.
- The pointer is the raw buffer of the array. A transposed or sliced view returns its parent's storage with non-trivial strides and an offset. Make it contiguous or walk the strides.
- `mlx_array_new_data` **copies** your buffer. For zero-copy, use `mlx_array_new_data_managed(_payload)` and keep the memory valid until the `dtor` runs, which can be long after you free your handle.

## The error handler

The default handler prints `MLX error: <msg> at <file>:<line>` and calls `exit(-1)`, so the process exits with status 255 and no crash report. **Every binding installs its own handler at load time.**

```c
static _Thread_local char mlx_last_error[1024];
static void on_mlx_error(const char* msg, void* data) {
  (void)data; snprintf(mlx_last_error, sizeof mlx_last_error, "%s", msg);
}
static void no_dtor(void* p) { (void)p; }
mlx_set_error_handler(on_mlx_error, &state, no_dtor);   // never NULL dtor with data
```

- **A NULL `dtor` silently drops your `data`.** With `mlx_set_error_handler(h, &state, NULL)`, the handler is called with `data == NULL`. To keep a pointer, pass a dtor, even a no-op one. With a real dtor, the data is freed when the handler is replaced.
- The handler is one global with no lock. Install it once, before any thread starts, and keep per-call state in thread-locals, as above. It runs on the thread that hit the error, before the failing function returns 1.
- After a nonzero status, treat the out-param as unspecified. Do not chain the next op on it.
- Many ops build lazily and succeed. Shape errors usually surface at the op, but kernel, allocation and stream errors surface at `mlx_eval` / `mlx_array_eval`, so check that status too.

## Closures and payloads

```c
mlx_closure cls = mlx_closure_new_func_payload(fn, payload, payload_free);
```

- The payload is held by a `shared_ptr` captured into every copy of the closure. That includes closures returned by `mlx_compile`, `mlx_value_and_grad`, `mlx_vjp` and the compile cache. **`payload_free` runs when the last copy dies, which can be after your `mlx_closure_free`.** Hand ownership to the dtor, and never free the payload yourself.
- A NULL dtor means "borrowed": the payload must outlive every derived closure and the compile cache.
- A nonzero return from `fn` becomes `mlx_closure returned a non-zero value`, delivered through the error handler of the caller of `mlx_closure_apply` or the transform. Keep your real message in the payload or in a thread-local.
- `fn` runs inside C++ frames. It must not unwind, longjmp or panic across them (Rust: `catch_unwind` inside the `extern "C" fn`).

## Threads and streams in a binding

Since mlx 0.31.2 a stream is **thread-affine**. Its command encoder is registered for the thread that created it. Build an array on one thread's default stream and evaluate it on another, and the op succeeds while `mlx_eval` fails with `There is no Stream(gpu, 0) in current thread.` (`see mlx-threads` for the core semantics).

| Binding shape | Stream to use |
| --- | --- |
| all MLX calls on one dedicated OS thread | default streams on that thread |
| calls from any thread behind your own lock (mlx-swift's model) | one `mlx_stream_new_thread_unsafe(dev)`; serialize every op and eval with a mutex |
| independent work per OS thread | `mlx_stream_thread_local_new(dev)`, resolved per thread with `mlx_stream_from_thread_local` |

- An **unevaluated** array is bound to the thread of its stream. An evaluated array's data can be read from any thread.
- `mlx_stream_new_thread_unsafe` gives no synchronization. Two threads evaluating on it at once is a data race.
- `mlx_clear_streams()` destroys the calling thread's streams. Call it when a worker thread exits.

## Rust, Go, Zig

**Rust**
- Wrap each handle in a newtype with `Drop` that calls `*_free`. Never `#[derive(Copy)]` or `#[derive(Clone)]`: implement `Clone` as `mlx_array_new` + `mlx_array_set`.
- Mark arrays `!Sync`. Mark them `Send` only if the binding uses thread-unsafe streams under a lock; otherwise unevaluated arrays cannot move threads.
- Every `extern "C" fn` callback (error handler, closure, dtor) wraps its body in `std::panic::catch_unwind`. Unwinding out of an `extern "C"` function aborts.
- Hand payloads over with `Box::into_raw`, with a dtor that does `drop(Box::from_raw(p))`.
- Convert `int` statuses into `Result` by reading the thread-local message the handler stored.
- Link `mlxc`, `mlx` and `c++`, plus the frameworks `Metal`, `Foundation`, `QuartzCore` and `Accelerate` for a static libmlx.

**Go**
- Goroutines migrate across OS threads, so the default streams break. Either run all MLX calls in one goroutine that called `runtime.LockOSThread()`, or use a single thread-unsafe stream behind a `sync.Mutex`.
- Finalizers and `runtime.AddCleanup` run on another goroutine. Either take the same mutex before `mlx_array_free`, or post the free to the owner goroutine. Prefer explicit `Free()` with `defer`.
- cgo passes the struct by value, so the copy-is-alias rule applies to every Go value holding one.

**Zig**
- `@cImport(@cInclude("mlx/c/mlx.h"))`. Put `defer mlx.mlx_array_free(x);` immediately after each `mlx_array_new`.
- Link `libmlxc.a`, `libmlx` and `c++`.
- Zig structs copy implicitly too, so store handles in single-owner containers.

## Building mlx-c

```bash
M=$(python -m mlx --cmake-dir)
cmake -S mlx-c -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLX_C_USE_SYSTEM_MLX=ON -DMLX_DIR=$M/share/cmake/MLX
cmake --build build                        # libmlxc.a + examples, no Metal Toolchain
clang -Imlx-c app.c build/libmlxc.a -L$M/lib -lmlx -lc++ -Wl,-rpath,$M/lib -o app
```

Without `MLX_C_USE_SYSTEM_MLX`, mlx-c fetches and source-builds mlx v0.32.4, which needs everything in the source-build section. `-DBUILD_SHARED_LIBS=ON` gives a shared `libmlxc`. The memory knobs (`mlx_set_cache_limit(&previous, bytes)` and friends) return the old value through the out-param (`see mlx-memory` for what the limits mean).
