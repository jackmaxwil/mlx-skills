# Authoring brief for mlx-skills

Shared rules for everyone writing a skill in this repo. Read it fully before writing.

## Model

This repo is the MLX/Metal counterpart of `~/Development/har-skills`. Before writing, read at least `har-skills/skills/har/SKILL.md`, `har-hot-path/SKILL.md` and `har-unsafe/SKILL.md` end to end and match their voice, density and shape:

- One file per skill: `skills/<name>/SKILL.md`. No `references/` folder, no scripts.
- Frontmatter has `name` and `description` only. The description lists the topics in a dash-separated run, then ends with "Load when ..." in the same style as har (`description: Rust runtime performance — measure before changing, ... Load before optimizing Rust or writing a frame, render, or parse loop.`).
- Length: 130–350 lines. Dense. Decision tables, wrong-vs-right code pairs, exact commands, exact error strings.
- A rule earns its line only if it makes an agent write different code. Do not explain what MLX is or tour the API.
- Normal technical English prose. Not terse "caveman" fragments.
- No topic is owned by two skills. If a topic belongs to another skill below, mention it in one clause with the skill name in backticks (`see mlx-transforms`) and move on.

## Accuracy

- Target versions: mlx 0.32.4, mlx-lm 0.32.2, mlx-swift 0.33.0, mlx-swift-lm 3.32.x, mlx-c 0.8.0, macOS 26.2+ (machine runs macOS 27.3). When behaviour changed recently, give the version (`since 0.31.2`).
- Upstream sources are cloned in `~/Development`: `mlx`, `mlx-lm`, `mlx-examples`, `mlx-swift`, `mlx-swift-lm`, `mlx-swift-examples`, `mlx-c`, `mlx-data`, `metal-cpp`, `metal-benchmarks`, `metal-flash-attention`. Every claim must come from that source, the official docs, or a run.
- Run every Python snippet you put in a skill that can run on this machine. The venv is `~/Development/mlx-skills/.venv` (mlx 0.32.4, mlx-lm 0.32.2, numpy, torch). Run with `~/Development/mlx-skills/.venv/bin/python script.py`, and keep test scripts in your scratchpad, not in the repo. The machine is an M4 Max (`applegpu_g16s`, so no M5 neural accelerators). `mx.fast.metal_kernel` works. The Xcode Metal Toolchain is installed (metalfe-32023.922.4, Xcode 27.2), so `xcrun metal`, extension builds and mlx-swift builds can be run. Anything you still cannot run, say "not run here" in your report only, not in the skill.
- If a snippet's real error or output differs from what you expected, the skill states what actually happens.
- No file:line references in the skills (they rot). Name the function or file only when the reader needs to go read it.

## Lineup and ownership

| Skill | Owns |
| --- | --- |
| `mlx` (base) | Lazy evaluation and eval placement, the implicit-eval list, the `async_eval` pipelining pattern, graph size, views pinning parent buffers, donation basics, dtype promotion and weak scalars, integer sum wrap, float64 CPU-only, TF32 default, numpy/torch/DLPack copy vs alias, lazy load, Module-as-dict traps (tuples, `_` prefix, freeze), recently changed APIs, PyTorch habits that break, upstream AGENTS.md contribution policy. |
| `mlx-verify` (base) | Correctness testing against references, per-dtype tolerances, adversarial test shapes (non-multiple of 32, <8 elements, transposed, broadcast, negative stride), benchmarking method (eval + synchronize, warmup, JIT first call), model-port verification (greedy token diff, test_models runner), CI shape. |
| `mlx-memory` (base) | Memory, cache and wired limits with real semantics and defaults, `sysctl iogpu.wired_limit_mb`, residency sets, measuring memory, preallocation and buffer reuse, `clear_cache` cadence, OOM and GPU-watchdog error decoder, "Resource limit exceeded". |
| `mlx-transforms` | `compile` (cache key, retrace triggers, state capture, fusion limits, shapeless rules), grad/value_and_grad/vjp/jvp/vmap, ordering of compile and grad, primitives without vjp/jvp/vmap, `custom_function`, `checkpoint`, `export_function`. |
| `mlx-threads` | Per-thread default streams, stream kinds, cross-thread errors, CPU/GPU overlap with `stream=`, per-thread RNG seeding, `clear_streams`, free-threaded Python. |
| `mlx-hot-path` | Profiling (Metal capture, xctrace, MLX_METAL_DEBUG), fast-path routing conditions (gemv/qmv/SDPA fallbacks), Apple GPU hardware table, command-buffer knobs, when not to write a kernel. |
| `mlx-kernel` | `mx.fast.metal_kernel` and C++/Metal extension authoring, signature generation, grid/threadgroup, templates, atomics, init_value, math_mode, vjp via custom_function, encoder contract, NAX gating, shader validation and printf. |
| `mlx-port` | Porting HF/PyTorch models into mlx-lm style: ModelArgs, sanitize, attention/mask/RoPE templates, cache choice, MoE routing, conv layouts, dtype discipline in models, shard/pipeline hooks. |
| `mlx-quant` | Quantization modes, group sizes, bits, predicates, recipes (mixed, dynamic_quant, DWQ, AWQ, GPTQ), activation quant, saved dtypes, GGUF, mlx-community publishing. |
| `mlx-train` | Training loops, compiled train-step template, LoRA/DoRA/full fine-tune, datasets, fuse, gradient accumulation, data parallel and FSDP. |
| `mlx-serve` | Generation loops, prefill chunking, KV and prompt caches, speculative decoding, chat templates, samplers, mlx-lm server flags, batching. |
| `mlx-distributed` | Backends (ring, JACCL, MPI, NCCL), RDMA setup, hostfiles, `mlx.launch`, `mlx.distributed_config`, collectives, tensor-parallel and pipeline sharding, `sharded_load`. |
| `mlx-swift-app` | Swift/Apple app integration: build and toolchain, metallib bundling as seen from Swift, concurrency and Sendable types, iOS/visionOS constraints, Swift token loop. Cross-reference the upstream `mlx-swift` and `mlx-swift-lm` skills instead of repeating their API tours. |
| `mlx-native` | mlx-c ownership and error handling, bindings from Rust/Go/Zig, C++ embedding with CMake, metallib search order, source builds and deployment targets, Metal Toolchain, CUDA/Linux deltas. |
