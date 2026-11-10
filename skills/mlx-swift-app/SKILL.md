---
name: mlx-swift-app
description: MLX in Swift apps and tools — SwiftPM vs xcodebuild vs the native build system, where default.metallib lands and how it is found, Metal Toolchain, submodules, package traits, version pinning, double-linked MLX, Linux SPM_CUDA, the error-handler lifecycle and task-local withError, Sendable types and MaterializedArray, keeping eval off the main actor and the cooperative pool, actor reentrancy, iOS and visionOS constraints (no simulator, entitlement, cache limit, backgrounding), and the Swift token loop. Load when adding mlx-swift or mlx-swift-lm to an app or CLI, when a Swift build or launch fails on metallib, Metal or Sendable errors, or when writing a Swift generation loop.
---

# MLX in Swift apps

The upstream repos ship API-tour skills: `mlx-swift/skills/mlx-swift` (arrays, ops, layers, optimizers, custom kernels) and `mlx-swift-lm/skills/mlx-swift-lm` (ModelContainer, ChatSession, generation API, tool calling, LoRA, porting). Install those for API shape. This skill covers what they leave out: the build, the launch, isolation, the device, and the loop.

## Build: which command produces a working binary

The README's "SwiftPM cannot build the Metal shaders" is stale. With Swift 6.x the default `swift build` system is Swift Build, which compiles the `.metal` files in `Cmlx`.

| Command | Compiles Metal | Result |
| --- | --- | --- |
| `swift build` / `swift run` (default `--build-system swiftbuild`) | yes | `.build/out/Products/<Config>/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` (about 4.7 MB), found from any working directory |
| `xcodebuild` / Xcode | yes | `Build/Products/<Config>/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib` for a CLI (it runs from a shell as is), or the same bundle inside the `.app` |
| `swift build --build-system native` (deprecated) | **no, silently** | builds and links, then dies at the first GPU op |
| either of the first two without the Metal Toolchain | — | `cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain` |

```bash
xcodebuild -showComponent MetalToolchain || xcodebuild -downloadComponent MetalToolchain
git submodule update --init --recursive     # only in a clone of mlx-swift itself
swift build -c release && .build/release/App
```

- **Never "fix" the toolchain error with `--build-system native`.** That trades a build error for a runtime exit.
- **A missing metallib at runtime** prints `MLX error: Failed to load the default metallib. library not found library not found ...` and exits with status 255. Typical causes: the executable was copied without `mlx-swift_Cmlx.bundle`, or it was built with the native system.
  - Fix by shipping the bundle next to the executable, or by setting `GPU.metallib = url` before the first MLX call.
  - The search order and override rules are in `mlx-native`.
- **CLIs built by `xcodebuild` as dynamic package frameworks** need `DYLD_FRAMEWORK_PATH=$BUILT_PRODUCTS_DIR/PackageFrameworks:$BUILT_PRODUCTS_DIR` when launched from a shell. mlx-swift-examples' `mlx-run` script does exactly this.
- **Error `The file "kernels" couldn't be opened`** in a clone of mlx-swift means the submodules are not checked out. SwiftPM consumers by URL do not need the submodule step.
- **mlx-swift-lm's own tests do not run under `swift test`.** Use `xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation`. For mlx-swift itself, CI runs `xcodebuild build-for-testing -scheme mlx-swift-Package -destination 'platform=macOS'` and then `xcrun xctest` on each `.xctest`.

### Dependencies and traits

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.33.0")),
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.32.3")),  // 3.32.3 pins mlx-swift 0.32.x; only main takes 0.33.0
.package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),   // for the MLXHuggingFace macros
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),  // Tokenizers
```

- **Use `.upToNextMinor`, not `from:`.** Minor versions break API, and mlx-swift-lm 3.x tracks one mlx-swift minor.
- **The examples pin older versions.** mlx-swift-examples pins mlx-swift 0.31.4 and mlx-swift-lm 3.31.3, so copied example code can call renamed APIs.
- **mlx-swift-lm 3.x no longer bundles a tokenizer or downloader.** The `#hubDownloader()`, `#huggingFaceTokenizerLoader()` and `#huggingFaceLoadModelContainer` macros compile only if the app itself depends on swift-huggingface (`HuggingFace`) and swift-transformers (`Tokenizers`), and imports both.
- **Traits:** `JACCL` (mlx-swift, macOS-only distributed backend, `see mlx-distributed`) and `FoundationModelsIntegration` (mlx-swift-lm, needs the OS 27 SDK) are on by default. Opt out with `traits: []` on the package reference. SwiftPM enables the **union** across the graph, so any other dependent that keeps the defaults turns the trait back on.
- **Never link MLX twice.** If App links MLX and App links YourFramework that also links MLX, SwiftPM gives each a static copy: two allocators, two eval locks, two error handlers. Make YourFramework a library, stop App linking MLX directly, or build MLX as frameworks from `xcode/MLX.xcodeproj` (prebuilt `.xcframework`s come from `tools/create-xcframework.sh`).
- **Minimum versions:** mlx-swift ≥ 0.32.3 (mlx-swift-lm requires it for an SDK issue). Ship iOS 26.2+ apps on 0.33.0, which fixes a launch crash in `Logger.isEnabled`; `MaterializedArray`/`MaterializedModule` are also new in 0.33.0. Package platform floors are macOS 14, iOS 17, tvOS 17 and visionOS 1.

### Linux

`swift build` on Linux selects the **CUDA** configuration unless `SPM_CUDA=0`.
- CPU-only needs `libopenblas`, `liblapack` and `gfortran` installed: `SPM_CUDA=0 swift build`.
- `MLXFast` and custom Metal kernels are excluded there.
- CUDA builds take `MLX_CUDA_INCLUDE_PATHS` (colon-separated roots for cudnn-frontend and CUTLASS) and `MLX_CCCL_DIR`.
- Wrap Apple-only calls (`GPU.metallib`, `MLXFast.metalKernel`, wired memory) in `#if canImport(Metal)`.

## Errors: the handler lifecycle

| State | An MLX error (shape mismatch, OOM, bad load) does |
| --- | --- |
| before the first `withError` / `withErrorHandler` anywhere in the process | mlx-c default: prints `MLX error: ...`, `exit(-1)` (status 255), no crash report |
| inside a `withError { }` on the **same task** | throws `MLXError.caught("...")` |
| after the handler is installed, outside any scope, or on another thread | `Fatal error: [broadcast_shapes] ...` (trap) |

The scope is a `@TaskLocal`. `DispatchQueue.async`, `Thread` and `concurrentPerform` workers do not inherit it, so an error there traps even inside an enclosing `withError`. mlx-swift-lm fixed this same crash in its weight loader.

```swift
// WRONG: the worker thread is outside the task-local scope -> fatalError
try withError { DispatchQueue.global().async { eval(load(shard)) } }

// RIGHT: open the scope where the MLX work runs
DispatchQueue.global().async {
    do { try withError { eval(load(shard)) } } catch { report(error) }
}
```

Use `try checkedEval(x)` where you would call `eval(x)` and want a throw. `eval(_ values: Any...)` traps on any value it cannot walk; it ignores `String` and numbers. Validate user-supplied shapes and paths before building the graph, because an error in a release iOS build is an exit or a trap, not an exception.

## Isolation and Sendable

| Type | Sendable | Cross an isolation boundary by |
| --- | --- | --- |
| `MLXArray` | no | `array.materialized()` / `materialize([a, b])` (one eval), or `asArray(Float.self)` |
| `MaterializedArray` | `@unchecked` | evaluated and sealed; `+=` or subscript-set through an `MLXArray`-typed variable traps |
| `Module` | no | `MaterializedModule(Model(...))`, built inline; the base is sealed, and mutating it later traps |
| `ModelContainer` | yes | `await container.perform { context in ... }`; return Sendable values |
| `ChatSession` | no | one per conversation or flow; never share across tasks |
| `Stream`, `Device`, `StreamOrDevice`, `DType` | yes | freely |

The compiler rejects the naive form: `non-Sendable type 'MLXArray' of let 'a' cannot exit main actor-isolated context`. Do not silence it with `nonisolated(unsafe)` or an `@unchecked` box. Materialize instead. `materialize()` inside `grad`, `vjp`, `vmap` or `compile` traps, so materialize **before** entering the transform (`see mlx-transforms`).

### Keep eval off the main actor and the cooperative pool

`eval`, `item`, `asArray` and every model forward block their thread until the GPU finishes.
- App targets built with `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` and Approachable Concurrency (as the mlx-swift-examples apps are) run an unannotated `func run() async` **on the main actor**, which freezes the UI.
- Blocking work on the global cooperative pool starves every other task.

mlx-swift-lm runs each generation on an actor with its own serial executor. Copy the pattern for your own loops:

```swift
actor MLXWorker {
    private nonisolated let queue = DispatchSerialQueue(label: "app.mlx")
    nonisolated var unownedExecutor: UnownedSerialExecutor { queue.asUnownedSerialExecutor() }
    func run<R: Sendable>(_ body: @Sendable () throws -> R) rethrows -> R { try body() }
}

let total = try await worker.run {
    try withError {
        let y = softmax(MLXArray(0 ..< 12, [3, 4]).asType(.float32), axis: -1).sum()
        try checkedEval(y)
        return y.item(Float.self)
    }
}
```

- **Keep `body` synchronous.** An `await` inside it hops off the serial queue.
- **Catch pool starvation in tests** by running with `LIBDISPATCH_COOPERATIVE_POOL_STRICT=1`.
- `@concurrent` (Swift 6.2) on a plain function only moves it to the cooperative pool. It fixes main-actor blocking, not pool starvation.

### Actors reenter; ModelContainer does not

A hand-written `actor Model { func generate() async { ... await ... } }` lets a second caller run between the first caller's suspension points. Two generations then interleave on one KV cache.
- mlx-swift-lm's `ModelContainer` wraps the context in `SerialAccessContainer`, an async mutex held for the whole `perform` closure. Route model access through it.
- For your own model types, either keep the actor's methods synchronous or hold an async lock across the whole operation.

### Threads, streams, randomness

- mlx-swift creates every stream with `mlx_stream_new_thread_unsafe` under its global `evalLock` and pools them, so tasks may hop threads freely. This differs from the C and C++ thread-affinity rule in `mlx-native`.
- `eval` schedules under `evalLock` and waits outside it (since 0.33.0), so evaluations from different tasks overlap.
- **Shared weights must be evaluated once, on one thread, before they are shared.** Lazy evaluation mutates the array.
- `Stream.withNewDefaultStream { }` and `withRandomState(state) { }` are task-local. They do not reach GCD workers.
- `compile` must list `MLXRandom.globalState` and any updated `Module` or `Optimizer` in `inputs:`/`outputs:`, or it freezes them at trace time (`see mlx-transforms`).

## iOS, iPadOS, visionOS

- **No Simulator.** It lacks the required MTLGPUFamily, and fails with asserts like `Dispatch Threads with Non-Uniform Threadgroup Size is not supported on this device`.
  - Develop on the "Mac (Designed for iPad)" destination or a multiplatform target, and test on a device.
  - Keep Simulator runs UI-only, behind `#if targetEnvironment(simulator)`. Do not chase these Metal errors.
- **Entitlement.** Add `com.apple.developer.kernel.increased-memory-limit` (the Increased Memory Limit capability) to any app that loads an LLM or a diffusion model. Every LLM and Stable Diffusion example ships it.
- **Cache limit.** Set `Memory.cacheLimit` once at launch: 20 MB for LLM chat (LLMEval, LLMBasic, MLXChatExample), 1 MB for diffusion on small-RAM devices. The default equals `Memory.memoryLimit`, which defaults above the working set (`see mlx-memory`). `Memory.memoryLimit` throttles evaluation; it does not prevent jetsam (`see mlx-memory`).
  - Size the model from `GPU.maxRecommendedWorkingSetBytes()`, not from `memoryLimit`. Pick parameters, quantization, `maxKVSize` and `kvBits` so weights plus KV fit well under it.
  - Use `Memory.snapshot()` before and after a turn to watch growth.
- **Deprecated APIs.** Use `Memory.*`, not the deprecated `GPU.*` properties. `Memory.withWiredLimit` (synchronous) is now a **no-op**. Use `WiredSumPolicy().ticket(size:kind:)` with `try await ticket.withWiredLimit { }` through `WiredMemoryManager.shared`.
- **Backgrounding kills GPU work.** A command buffer submitted after the app leaves the foreground fails with `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`. Cancel the generation task when `scenePhase` leaves `.active`, and make the loop check `Task.isCancelled` **before** queuing the next step.
- **First use is slow.** Most kernels are JIT-compiled from embedded source the first time they run, and the system caches them afterwards. Run a short warm-up generation behind the loading UI.

## The Swift token loop

With mlx-swift-lm, use `generate`/`generateTask` (`see mlx-serve` for sampling, caches and speculative decoding). Await the returned task after breaking out of a stream early, so cleanup and the final synchronize run. A hand-written loop needs every line of this:

```swift
func decodeLoop(
    first: MLXArray, maxTokens: Int,
    step: (MLXArray) -> (token: MLXArray, state: [MLXArray]),
    emit: (Int) -> Void
) {
    var y = first
    for i in 0 ..< maxTokens {
        if Task.isCancelled { break }          // before step: no new GPU work after cancel
        let id: Int = autoreleasepool {
            let previous = y
            let (token, state) = step(previous)
            asyncEval([token] + state)          // queue step i+1 ...
            y = token
            if i % 256 == 0 { Memory.clearCache() }
            return previous.item(Int.self)      // ... while reading step i
        }
        emit(id)
    }
    Stream.defaultStream.synchronize()          // settle the last asyncEval before returning
}
```

| Line | What breaks without it |
| --- | --- |
| `autoreleasepool` per step | hundreds of autoreleased Metal wrappers per forward; host memory grows without bound over a long generation |
| cache state in the `asyncEval` | the functional cache updates chain and pin every earlier step's intermediates |
| `Memory.clearCache()` every 256 steps | odd and growing buffer sizes accumulate in the pool |
| cancellation check before `step` | one extra command buffer after cancel or backgrounding |
| final `synchronize()` | process exit or teardown while work is in flight hits scheduler asserts |

- **Emit off the worker.** Run the loop inside `MLXWorker.run`, and deliver tokens through an `AsyncStream` continuation. Do not hop to the main actor inside the loop.
- **Prefill** in chunks. Call `asyncEval(cache)` after each chunk and one `eval(cache)` at the end, so the CPU builds chunk N+1 while the GPU runs chunk N. mlx-swift-lm clears the cache after each prefill chunk and on the first generated token.
- **Speculative decoding** in `ChatSession` falls back to ordinary decoding without raising an error when temperature is nonzero, the input has media, or the draft model would exceed the memory budget (3.32.3 falls back only for media and memory; the temperature case and `SpeculativeDecodingFallbackReason` are on main). Read the fallback reason before you conclude that speculation is slow.
