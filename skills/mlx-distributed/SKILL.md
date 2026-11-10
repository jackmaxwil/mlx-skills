---
name: mlx-distributed
description: MLX across more than one process or Mac — backend choice (ring, JACCL, MPI, NCCL), RDMA over Thunderbolt 5 setup, hostfile schema, mlx.launch and mlx.distributed_config, init(strict=True), what each backend cannot do, collectives and send/recv, tensor-parallel shard_linear, fully_shard, pipeline stages with mx.depends, sharded_load, MLX_METAL_FAST_SYNCH. Load when work spans more than one process or more than one Mac.
---

# Distributed MLX

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. Data-parallel training loops, `average_gradients` and the FSDP training recipe belong to `mlx-train`; this skill owns the communication layer underneath them.

## The silent single process

```python
world = mx.distributed.init()          # launched with plain `python`: size 1, no error
x = mx.distributed.all_sum(x)          # a no-op on a size-1 group
```

Every collective (`all_sum`, `all_gather`, `all_max`, `all_min`, `sum_scatter`) returns its input unchanged on a group of size 1, so a script started without `mlx.launch`, or with a broken environment, runs to completion on one machine and reports single-machine numbers. When more than one process is required, say so:

```python
world = mx.distributed.init(strict=True)   # RuntimeError instead of a size-1 group
assert world.size() == expected_size
```

| Call | Without a launcher |
| --- | --- |
| `init()` / `init(backend="any")` | size-1 group |
| `init(strict=True)` | `RuntimeError: [distributed] Couldn't initialize any backend` |
| `init(strict=True, backend="ring")` | `[ring] You need to provide via environment variables both a rank (MLX_RANK) and a hostfile (MLX_HOSTFILE) ...` |
| `init(strict=True, backend="jaccl")` | `[jaccl] You need to provide via environment variables a rank (JACCL_RANK/MLX_RANK), a device file ... and a coordinator ip/port ...` |
| `send(x, 0)` on a size-1 group | `ValueError: Cannot send to a singleton group` — unlike collectives, `send`/`recv` do **not** no-op |

`backend="any"` tries NCCL (only when CUDA is present), then ring, then MPI, then JACCL. The first backend that initializes is cached under `"any"`, so a later `init()` returns it even if you meant another. A failed attempt is not cached and can be retried.

`mx.distributed.is_available("jaccl")` is `True` on any macOS that ships `librdma.dylib`. It says the library loads, not that RDMA is enabled or cabled.

## Choosing a backend

| Backend | Transport | Use for | Cannot do |
| --- | --- | --- | --- |
| `ring` | TCP sockets, Ethernet or Thunderbolt IP | default on Macs; data-parallel `all_sum` / `all_gather`; pipeline stages | `send`/`recv` except to the rank either side; `sum_scatter` (so no `fully_shard` backward; FSDP: `see mlx-train`); `Group.split` |
| `jaccl` | RDMA over Thunderbolt 5, macOS 26.2+ | tensor parallelism and anything latency-bound (an order of magnitude lower latency than ring) | `Group.split` |
| `jaccl-ring` | the same, ring topology | Macs cabled only to their two neighbours | `Group.split` |
| `mpi` | MPI over TCP | existing MPI clusters | `sum_scatter` |
| `nccl` | NCCL, CUDA only | Linux/CUDA builds of MLX; the `mlx.launch` default when CUDA is present | `send`/`recv` (`[nccl] Send not supported in NCCL backend.`), so no pipeline stages |

Two consequences that change code:

- `fully_shard` reduce-scatters its gradients, so its backward pass fails on ring and MPI with `[ring] sum_scatter not supported.` (`[mpi] sum_scatter not yet implemented.`) The forward pass works, which means the error appears at the first `mx.eval` of the gradients, not at wrap time. Use JACCL or NCCL for FSDP.
- Without `Group.split`, a Mac cluster has one group. `sharded_load` accepts a `pipeline_group` and a `tensor_group`, but on ring or JACCL you can pass the world to one of them, not split it into both.

Communication on ring, JACCL and MPI runs on the **CPU** stream (NCCL on the GPU), so each collective in the middle of GPU work is a CPU–GPU synchronization point.

## Launching

`mlx.launch` runs the command it is given verbatim on each host. A bare script path is executed, not interpreted:

```
mlx.launch -n 2 train.py                                   # exec: train.py: not found (exit 127)
mlx.launch -n 2 --python "$(which python)" train.py        # local test, 2 ranks on 127.0.0.1
mlx.launch --hostfile hosts.json --python /path/to/venv/bin/python train.py
mlx.launch --hostfile hosts.json -- python -m mlx_lm chat --model mlx-community/Qwen3-30B-A3B-4bit
```

`--python` must be a path that exists on **every** host. Keep the same macOS build, Python and mlx version on all nodes.

| Flag | Effect |
| --- | --- |
| `-n N` | repeat each host N times (local multi-process testing) |
| `--hosts ip1,ip2` | ad-hoc host list; ring needs **IPs** here (`The ring backend requires IPs to be provided instead of hostnames`) |
| `--hostfile f.json` | hosts, and optionally the backend and env vars |
| `--backend ring\|jaccl\|jaccl-ring\|mpi\|nccl` | overrides the hostfile; default is the hostfile's, else `nccl` with CUDA, else `ring` |
| `--env K=V` | repeatable; appended after the hostfile's `envs` |
| `--starting-port`, `-p` | first port (default 32323); ring uses one port per rank per IP |
| `--connections-per-ip` | extra ring sockets per IP |
| `--cwd`, `--verbose` | remote working directory; `--verbose` also sets `MLX_RING_VERBOSE=1` |

`mlx.launch` forwards stdin to every rank and merges their stdout, so `pdb` and interactive `mlx_lm chat` work. A rank that exits non-zero shows as `[WARN] Node with rank R exited with code C`; treat any `[WARN]` line as a failure in CI.

Without the launcher (a cluster scheduler starts the processes), set the environment yourself:

| Backend | Required environment |
| --- | --- |
| ring | `MLX_RANK`, `MLX_HOSTFILE` = path to `[["ip:port", ...], ...]`, one list per rank |
| jaccl | `MLX_RANK`, `MLX_IBV_DEVICES` = path to the RDMA device matrix, `MLX_JACCL_COORDINATOR` = `ip:port` that rank 0 listens on; `MLX_JACCL_RING=1` for ring topology. `JACCL_*` spellings are also accepted |
| nccl | `MLX_RANK`, `MLX_WORLD_SIZE`, `NCCL_HOST_IP`, `NCCL_PORT`, `CUDA_VISIBLE_DEVICES` |

## Hostfiles

```json
{
  "backend": "jaccl",
  "envs": [],
  "hosts": [
    {"ssh": "mac-1", "ips": ["10.0.0.1"], "rdma": [null, "rdma_en5", "rdma_en4"]},
    {"ssh": "mac-2", "ips": [],           "rdma": ["rdma_en5", null, "rdma_en3"]},
    {"ssh": "mac-3", "ips": [],           "rdma": ["rdma_en4", "rdma_en3", null]}
  ]
}
```

- Rank is position in `hosts`. A bare list of host objects is also accepted.
- `ssh` is what `ssh` connects to without a password prompt; `ips` is where ring listens.
- **Ring:** every host needs at least one IP, and every rank must have the same number of connections left and right. Several IPs per host (one per Thunderbolt cable) become several sockets.
- **JACCL:** only rank 0 needs an IP; it is the coordinator. `rdma[j]` names the device that connects this host to host `j`, and the diagonal must be `null`. For two cables between a pair, write a list (`["rdma_en2", "rdma_en3"]`); position `k` must be cabled to position `k` on the peer. With uneven cable counts JACCL uses the smallest count for every pair.
- JACCL mesh needs a **direct cable between every pair**. `jaccl-ring` needs only neighbours. The launcher rejects gaps before anything starts: `The hostfile is malformed: no RDMA device is listed for b to c, c to b`.

## RDMA over Thunderbolt 5

One-time per Mac. It **cannot** be done remotely, even with sudo:

1. Boot into macOS Recovery, open Utilities → Terminal, run `rdma_ctl enable`, reboot.
2. Check: `ibv_devices` lists `rdma_en*` devices.
3. Disable the Thunderbolt Bridge interface, and give each cable its own isolated subnet. RDMA does not use TCP/IP, but setup still requires both.

`mlx.distributed_config` does steps 3 and the hostfile for you:

```
mlx.distributed_config --verbose --hosts mac-1,mac-2,mac-3 --over thunderbolt --dot | dot -Tpng | open -f -a Preview
mlx.distributed_config --verbose --hosts mac-1,mac-2,mac-3 --over thunderbolt \
    --backend jaccl --auto-setup --output-hostfile hosts.json
```

The `--dot` run is how you find a miscabled pair before a job hangs. `--auto-setup` needs passwordless sudo on every node; without it the tool prints the commands to run. `--backend` takes `ring`, `jaccl` or `jaccl-ring`; `--over ethernet` writes a ring hostfile from the hosts' IPs.

## `MLX_METAL_FAST_SYNCH`

It switches CPU–GPU fences to a faster path, which matters because communication runs on the CPU. It is off by default and the docs say to leave it unset: it can deadlock and leave the GPU wedged (ml-explore/mlx#3142). Older guides, talks and the hostfile docstring still pass `--env MLX_METAL_FAST_SYNCH=1`; do not copy that into a default config. If you try it, A/B it on the exact workload and keep a way to kill the job. It is read once per process, so set it at launch, not from Python mid-run.

## Collectives and point-to-point

All distributed ops are lazy graph nodes like any other op: nothing is sent until the result is evaluated.

| Op | Notes |
| --- | --- |
| `all_sum`, `all_max`, `all_min` | elementwise across ranks; any dtype, including `bool` |
| `all_gather(x)` | concatenates along axis 0 |
| `sum_scatter(x)` | axis 0 must divide by the group size; JACCL and NCCL only |
| `send(x, dst)`, `recv(shape, dtype, src)`, `recv_like(x, src)` | ring: neighbours only, `[ring] Send only supported to direct neighbors` |

Batch small reductions into one: many small `all_sum` calls are latency-bound. `nn.average_gradients` packs a gradient tree into ~32 MiB messages for this reason (`mlx-train`).

**A `send` whose result is never evaluated never happens.** The receiver then blocks until the peer process exits and fails with `[ring] connection to a peer was lost`:

```python
# Wrong: the returned array is dropped, the graph prunes the send.
mx.distributed.send(h, dst)
return logits

# Right: tie the send into something that is evaluated.
h = mx.distributed.send(h, dst)
cache[-1].keys = mx.depends(cache[-1].keys, h)
```

A barrier is a tiny CPU reduction: `mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))`. mlx-lm runs one after loading so that slow loaders do not time out the others.

**Ranks do not share a random seed.** Each process seeds from its own clock, so `nn.Linear(...)` initializes differently on every rank, and a sharded layer silently computes garbage. Call `mx.random.seed(s)` with the same `s` on every rank before building a randomly initialized model, or load weights from a file.

## Tensor parallelism

```python
from mlx.nn.layers.distributed import shard_linear

attn.q_proj = shard_linear(attn.q_proj, "all-to-sharded", group=g)   # also k, v, gate, up
attn.o_proj = shard_linear(attn.o_proj, "sharded-to-all", group=g)   # also down
attn.n_heads //= g.size()
attn.n_kv_heads //= g.size()
```

- `all-to-sharded` splits output features and all-sums gradients in the backward pass. `sharded-to-all` splits input features and all-sums the forward output, so it expects its **input already sharded**. A pair of them equals the unsharded pair to float rounding.
- `shard_linear` returns a new layer and works on `Linear` and `QuantizedLinear`. `shard_inplace` changes only the parameters, so the module itself must already perform the communication.
- Divide every head count by the group size. Forgetting gives a reshape error or wrong output. The group size must divide `n_kv_heads`; with GQA, 8 KV heads cap tensor parallelism at 8 ranks.
- Fused projections need `segments`: `segments=3` for a fused QKV, or a list of split points for uneven parts, so that each rank gets its slice of each part rather than a slice of the concatenation.
- Quantized layers shard along whole quantization groups. A `sharded-to-all` `QuantizedLinear` needs `input_dims % (group_size * N) == 0`, otherwise `[quantize] The last dimension of the matrix needs to be divisible by the quantization group size 64`.
- Tensor parallelism communicates twice per layer per token. Over ring TCP that latency usually erases the gain; use JACCL.

## Pipeline parallelism

mlx-lm models that subclass `PipelineMixin` expose `model.model.pipeline(group, split=None)`:

- Rank `size-1` runs the **first** layers and rank 0 the **last**; the low ranks get the extra layers in an uneven split. Pass `split=[...]` with positive per-rank layer counts that sum to the layer count to rebalance.
- Each stage does `recv_like` from `rank + 1`, runs its layers, `send`s to `rank - 1` tied in with `mx.depends`, and finally `all_gather(h)[:B]` so every rank holds the output. Only neighbours talk, so pipeline works on ring.
- Guard the point-to-point calls with the stage index. `send`/`recv` throw on a size-1 group, unlike collectives.

## mlx-lm entry points

```python
from mlx_lm.utils import sharded_load
model, tokenizer = sharded_load(repo, pipeline_group=None, tensor_group=None)
```

- With neither group given it picks tensor parallelism if the model has `shard()`, otherwise pipelining, over `mx.distributed.init()`.
- In pipeline mode each rank downloads only the safetensors files its layers live in, found through `model.safetensors.index.json`. A checkpoint missing layers fails with `The checkpoint has no weights for layers [...]. Convert the model with mlx_lm.convert first.`
- `mlx_lm chat`, `mlx_lm server` and `mlx_lm.benchmark` take `--pipeline` to choose pipelining over the default tensor parallelism. Adapters and draft models are refused in distributed mode.

## Local test before a cluster

Two ranks on one Mac exercise the ring code paths, including the backend limits above:

```
mlx.launch -n 2 --python "$(which python)" test_dist.py
```

`init(strict=True)`, assert `world.size()`, seed every rank identically, compare a sharded layer against the unsharded one, and run a backward pass if the job trains. JACCL-only and RDMA-only behaviour cannot be reproduced this way.
