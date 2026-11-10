# MLX Skills

Fourteen skills that teach AI coding agents to use MLX and Metal on Apple silicon correctly. They cover where evaluation happens, what memory actually costs, how kernels are dispatched, and how to prove a port or a kernel is right.

Each skill is a single dense `SKILL.md` made of decision tables, wrong-versus-right pairs, exact commands and real error strings. No topic belongs to two skills; a skill points to another one instead of repeating it. The snippets were run against mlx 0.32.4 and mlx-lm 0.32.2 on macOS 27.

## Install

```bash
npx skills add jackmaxwil/mlx-skills
```

Install one skill, or a few:

```bash
npx skills add jackmaxwil/mlx-skills --skill mlx-kernel
```

Or use one without installing it:

```bash
npx skills use jackmaxwil/mlx-skills@mlx-verify
```

Installs into Claude Code, Cursor, Codex, and every other agent the [skills CLI](https://github.com/vercel-labs/skills) supports.

## The skills

| Skill | Load when |
| --- | --- |
| `mlx` | Writing or reviewing any MLX code: lazy evaluation and where to call `mx.eval`, the implicit evals, `async_eval` pipelining, dtype promotion and weak scalars, views and copies, NumPy and PyTorch interop, `Module` traps, recently changed APIs, upstream contribution policy. |
| `mlx-verify` | Deciding how to test or benchmark an MLX change: reference comparisons and per-dtype tolerances, adversarial shapes and views, gradient checks, honest benchmarks, port verification, CI. |
| `mlx-memory` | A model is large, memory looks like a leak, or the GPU reports out of memory or a timeout: memory, cache and wired limits, residency, measurement, buffer reuse, error decoder. |
| `mlx-transforms` | Using `mx.compile`, `grad`, `vjp`, `jvp` or `vmap`: cache keys and retrace triggers, state capture, fusion, shapeless rules, compile/grad ordering, custom functions, checkpointing, export. |
| `mlx-threads` | MLX runs on more than one thread, or CPU and GPU work should overlap: per-thread streams, stream kinds, cross-thread errors, random seeding, free-threaded Python. |
| `mlx-hot-path` | Before optimizing: profiling and GPU capture, which shapes hit MLX's fast kernels, Apple GPU hardware facts, command-buffer tuning, when not to write a kernel. |
| `mlx-kernel` | Writing `mx.fast.metal_kernel` or a C++/Metal extension: signature generation, grid and threadgroup, templates, atomics, VJPs, the encoder contract, neural-accelerator gating, shader debugging. |
| `mlx-port` | Porting a Hugging Face or PyTorch model: config, `sanitize`, attention, masks and RoPE, caches, MoE routing, conv layouts, dtype discipline, sharding hooks. |
| `mlx-quant` | Quantizing or converting a model: modes, group sizes and bits, predicates, recipes from mixed-bit to DWQ, AWQ and GPTQ, saved dtypes, GGUF, publishing. |
| `mlx-train` | Writing a training loop or fine-tuning: the compiled train step, LoRA, DoRA and full fine-tunes, datasets, fusing adapters, gradient accumulation, data parallel and FSDP. |
| `mlx-serve` | Building generation or a server: prefill, KV and prompt caches, speculative decoding, chat templates, samplers, server flags, batching. |
| `mlx-distributed` | Work spans more than one Mac: ring, JACCL over Thunderbolt RDMA, MPI and NCCL, hostfiles, `mlx.launch`, collectives, tensor and pipeline parallelism. |
| `mlx-swift-app` | Shipping MLX in a Swift, iOS or visionOS app: toolchain and metallib bundling, Swift concurrency and `Sendable`, device memory, backgrounding, the Swift token loop. |
| `mlx-native` | Embedding MLX from C or C++, or binding it from Rust, Go or Zig: mlx-c ownership and errors, CMake, metallib search order, source builds, deployment targets, CUDA and Linux. |

`mlx`, `mlx-verify`, and `mlx-memory` are the base three. The rest load on demand.

For the Swift API itself, use the official [`mlx-swift`](https://github.com/ml-explore/mlx-swift/tree/main/skills/mlx-swift) and [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm/tree/main/skills/mlx-swift-lm) skills; `mlx-swift-app` covers what they leave out.

## Sources

These skills are original prose distilled from the MLX source and from published work by others. The thinking is theirs; any errors in compression are mine.

- **The MLX source and docs**: [ml-explore/mlx](https://github.com/ml-explore/mlx), [mlx-lm](https://github.com/ml-explore/mlx-lm), [mlx-swift](https://github.com/ml-explore/mlx-swift), [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) and [mlx-c](https://github.com/ml-explore/mlx-c), read at their November 2026 heads. Also the [MLX documentation](https://ml-explore.github.io/mlx/build/html/), especially lazy evaluation, compile, distributed, custom Metal kernels and the Metal debugger, and the mlx-lm `LORA.md`, `SERVER.md` and `LEARNED_QUANTS.md`. Most of every skill comes from here.
- **[Making MLX Go Fast](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)**, Awni Hannun, with his [`fast-mlx`](https://github.com/awni/mlx-skills) skill. This is the practitioner's view of evaluation, dtypes and compile behind `mlx` and `mlx-transforms`.
- **[Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)**, Apple Machine Learning Research, which is where `mlx-hot-path` gets its prefill-versus-decode bounds.
- **WWDC sessions on MLX**: [Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/315/) and [Explore LLMs on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/298/) (WWDC25), and [Run local agentic AI on the Mac using MLX](https://developer.apple.com/videos/play/wwdc2026/232/) and the [distributed MLX session](https://developer.apple.com/videos/play/wwdc2026/233/) (WWDC26). Behind `mlx-serve` and `mlx-distributed`.
- **Metal 4 and Metal for ML**: [Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/), [Combine Metal 4 machine learning and graphics](https://developer.apple.com/videos/play/wwdc2025/262/), [Optimize custom ML operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/), the [M5 and A19 GPU tech talk](https://developer.apple.com/videos/play/tech-talks/111432), the [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) and the [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf). Behind `mlx-kernel` and `mlx-hot-path`.
- **[metal-benchmarks](https://github.com/philipturner/metal-benchmarks)** and **[metal-flash-attention](https://github.com/philipturner/metal-flash-attention)**, Philip Turner, for Apple GPU microarchitecture: instruction throughput, register pressure, bf16 handling across generations. These measurements stop at Apple8 (M2), and the skills say so where it matters.
- **[Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375)** for dynamic caching and occupancy on Apple9 and later.
- **[mlx-community](https://huggingface.co/mlx-community)** on Hugging Face for the conventions behind quantized-checkpoint publishing in `mlx-quant`.
- **Apple's [game-porting-skills](https://github.com/apple/game-porting-toolkit)**, prior art for Metal agent skills on the graphics side. This family covers compute and ML and does not overlap it.

## Contributing

A change should make an agent write different code. Rules that only describe MLX do not earn their lines. Keep the ownership boundaries in [AUTHORING.md](AUTHORING.md): if a topic belongs to another skill, reference it in a clause rather than restating it. Run every snippet you add.

These skills are for using MLX. Upstream `ml-explore` repositories do not accept agent-written pull requests, commit messages or review replies; see their `AGENTS.md`.
