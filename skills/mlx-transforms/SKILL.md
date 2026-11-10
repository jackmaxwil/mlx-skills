---
name: mlx-transforms
description: MLX function transforms — mx.compile cache key and retrace triggers, state capture with inputs/outputs, random state in compiled code, fusion limits, shapeless rules, grad/value_and_grad/vjp/jvp/vmap contracts, ordering of compile with grad and vmap, primitives without vjp/jvp/vmap, custom_function, checkpoint, export_function and import_function. Load before writing or reviewing mx.compile, a gradient, a vmap, a custom VJP, or an exported function.
---

# MLX transforms

Written against **mlx 0.32.4**. Each error string below was produced on that version. Lazy evaluation basics are in `mlx`; the training-loop template is in `mlx-train`.

## Order: compile outermost

The `Compiled` primitive has no vjp, jvp or vmap. When a compiled function receives tracer inputs, from an enclosing `grad` or `vmap`, it skips compilation and runs the Python body each call:

| You write | Python body runs | Compiled |
| --- | --- | --- |
| `mx.grad(mx.compile(loss))` | every call | no |
| `mx.compile(mx.grad(loss))` | once | yes |
| `mx.vmap(mx.compile(f))` | every call | no |
| `mx.compile(mx.vmap(f))` | once | yes |

Measured over three calls each on 0.32.4. Compile the whole step: forward, grad and optimizer update together.

## Compile: what causes a retrace

The cache is keyed per Python function object (`id(fun)`) and holds a list of entries. An entry matches on: the calling thread's default stream, the `shapeless` flag, every input array's shape (only `ndim` when shapeless) and dtype, the pytree structure of the arguments, and the **value** of every non-array argument.

| Pattern | Traces | Fix |
| --- | --- | --- |
| `cf(x, k)` with a Python `int` that changes | 5 in 5 calls | pass `mx.array(k)` |
| `cf(x, s)` with a Python `float` that changes | one per distinct value | pass `mx.array(s)` |
| `mx.compile(f)(x)` inline in a loop | 5 in 5 calls | compile once, outside the loop |
| `mx.compile(lambda x: ...)` in a loop | 5 in 5 calls | a new lambda is a new id |
| input shapes that vary per call | one per new shape | bucket or pad shapes, or `shapeless=True` |
| `with mx.stream(mx.cpu): cf(x)` after a GPU call | +1 | expected: the stream is part of the key |
| same compiled function on a new thread | +1 per thread | caches are per thread (`mlx-threads`) |

- Entries are never evicted while the compiled function lives, so a server that sees many sequence lengths keeps one graph per length. Destroying the compiled function erases its entries on the thread that last called it.
- Non-array arguments must be `int`, `float`, `str`, `None`, or lists, tuples or dicts of those and arrays. Anything else raises `[compile] Function arguments must be trees of arrays or constants (floats, ints, strings, or None), but received type object.`
- `mx.disable_compile()`, or setting `MLX_DISABLE_COMPILE` to any value (even `0`), turns `mx.compile` into the identity. Use it to check whether a bug lives in compile.
- On the CPU device, compile needs a working system C++ compiler (the CPU backend JIT-compiles C++). Without one it silently runs uncompiled.

## Compile: state is frozen unless captured

Arrays the function reads from a closure, a global, or `self` are **graph constants**. They are frozen at trace time:

```python
w = mx.array(1.0)
cf = mx.compile(lambda x: x * w)
cf(x)                       # 1.0
w = mx.array(5.0)
cf(x)                       # still 1.0

fwd = mx.compile(model)     # weights frozen into the graph
model.update(new_params)    # or load_weights(...)
fwd(x)                      # old weights, and the old buffers stay alive
```

Capture mutable state explicitly. `inputs=` is read on every call and `outputs=` is written back on every call:

```python
from functools import partial

state = [model.state, optimizer.state, mx.random.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

loss = step(x, y)
mx.eval(loss, state)
```

- **Randomness.** Without `mx.random.state` in both `inputs` and `outputs`, the key is a constant: `mx.compile(lambda: mx.random.uniform(shape=(2,)))` returns the same values on every call. Dropout inside a compiled step needs it.
- **Leaking a tracer.** Appending an intermediate to an outside list, or assigning one to an attribute, without listing that container in `outputs=` leaves a placeholder behind. Evaluating it raises `[eval] Attempting to eval an array without a primitive. If you are compiling a function, make sure all the inputs and outputs are captured`.
- The Python body runs only while tracing. `print`, counters, logging and assertions inside it fire on the first call and on retraces, never on cache hits.
- No value reads inside compiled code. `x.item()`, `if x.sum() > 0:` and `np.array(x)` raise `[eval] Attempting to eval an array during function transformations like compile or vmap is not allowed.` Use `mx.where`, or decide outside the compiled function.

## Compile: what fuses

Only elementwise work fuses: unary ops, binary ops, `where`, `astype` and broadcasts on one stream. A fused kernel takes at most 24 input arrays and spans at most 11 ops in depth. Matmul, reductions, gather and scatter, slicing, concatenate, sort and every `mx.fast` op are fusion boundaries. Compile still removes Python overhead and does common-subexpression elimination across them.

Expect gains from activation chains, norms written by hand, loss arithmetic and optimizer updates. A function that is one big matmul gains nothing from fusion. The first call of each new trace JIT-compiles Metal source; exclude it from timing (`mlx-verify`).

## Shapeless compile

`shapeless=True` keeps one trace across shape changes. A change in `ndim` or dtype still retraces. Every primitive in the graph must infer its output shape from its inputs. These cannot, and raise at the first call:

| Op | Error | Shapeless alternative |
| --- | --- | --- |
| `x[:n]`, `x[a:b]` | `[Primitive::output_shapes] Slice cannot infer output shapes.` | `mx.slice(x, mx.array([a]), axes=(0,), slice_size=(n,))` |
| `mx.pad` | `Pad cannot infer output shapes.` | pad outside the compiled function |
| `mx.split` | `Split cannot infer output shapes.` | `mx.slice` per piece |
| `mx.random.*` | `RandomBits cannot infer output shapes.` | draw the noise outside and pass it in |

Scatter, FFT, `as_strided`, `view`, `qr`/`svd`/`inv`/`cholesky`/`lu` and the fused VJP kernels are also excluded. Scan ops are supported since 0.32.3.

**Shape arithmetic is baked in, sometimes silently.** A `reshape` keeps its trace-time target:

```python
f = mx.compile(lambda x: x.reshape(x.shape[0] * x.shape[1], -1), shapeless=True)
f(mx.ones((2, 3, 4)))     # (6, 4)
f(mx.ones((5, 5, 3)))     # ValueError: [reshape] Cannot reshape array of size 75 into shape (6,12).
f(mx.ones((4, 3, 2)))     # (6, 4): silently wrong, should be (12, 2)

g = mx.compile(lambda x: x.flatten(0, 1), shapeless=True)   # right: (6, 4) then (25, 3)
```

Use `flatten`, `unflatten`, `-1` relative to dims that do not change, and broadcasting instead of reading `x.shape`. Then test a shapeless function on at least two shapes whose sizes collide (`mlx-verify`).

## grad and value_and_grad

| Contract | Behaviour |
| --- | --- |
| return value | a scalar array. A shape-`(3,)` output raises `[vjp] Output shape (3) does not match cotangent shape ().` |
| auxiliary outputs | `value_and_grad(f)` accepts `(scalar, aux...)` and returns `((loss, aux...), grads)`. Plain `grad` rejects a tuple |
| `argnums` | defaults to `0`. A tuple returns a tuple of grads. Each argument may be a tree; the grad has the same structure |
| `argnames="w"` | returns `(grads_for_argnums_or_None, {"w": grad})` |
| unused argument | a dense `zeros_like` gradient, not `None` |
| gradient dtype | the primal's dtype. bf16 parameters get bf16 gradients |
| integer input | **no error**: `grad(lambda x: (x*x).sum())(mx.array([1, 2]))` returns `int32` `[2, 4]`. Cast inputs to float |
| index arrays | `[gather] Cannot calculate VJP with respect to indices. Use stop_gradient on indices to stop gradients from being computed.` Keep indices out of `argnums` |

- `mx.stop_gradient(x)` is identity in the forward pass and cuts the backward pass. There is no `detach()`.
- `x.item()` and Python `if` on values work inside `grad`, which evaluates the forward pass, but they are syncs and break once the step is compiled.
- The forward graph stays alive until the gradients are evaluated. Use `mx.checkpoint` to trade compute for memory (below).
- `nn.value_and_grad(model, fn)` differentiates `model.trainable_parameters()` and calls `fn(*args)`. Frozen and `_`-prefixed arrays get no gradient (see `mlx` for the Module rules).

`mx.vjp(f, primals, cotangents)` and `mx.jvp(f, primals, tangents)` take and return lists. `mx.vmap(f, in_axes=..., out_axes=...)` maps over positional arguments only: keywords raise `[vmap] Keyword arguments are not supported.`

## Primitives without transforms

The default error is `[Primitive::vjp] Not implemented for <Name>.` (and likewise `jvp` and `vmap`):

| Missing | Primitives (op that creates them) |
| --- | --- |
| vjp, jvp | every `mx.linalg` decomposition: `QRF`, `SVD`, `Inverse`, `Cholesky`, `Eig`, `Eigh`, `LUF` (they also need `stream=mx.cpu`); `GatherQQMM`; `SegmentedMM`; `Load`; `RandomBits`; `View`; `BitwiseInvert`; `NumberOfElements` |
| jvp | `Convolution`, `GatherMM`, `BlockMaskedMM`, `Depends` |
| vmap | `GatherMM`, `BlockMaskedMM`, `SegmentedMM`, `AsStrided`, `Depends`, `GatherQQMM`, `QRF`, `LUF`, `Load` |
| partial vjp | `QuantizedMatmul`: no gradient for the packed weights, and none for scales outside `affine` mode; `Scatter` max/min reductions; masks of `BlockMaskedMM` and `masked_scatter` |

`mx.grad` through `mx.linalg.inv` raises `[Primitive::vjp] Not implemented for Inverse.` Write the gradient with `custom_function` instead.

## custom_function

```python
@mx.custom_function
def mul(x, y):
    return x * y

@mul.vjp
def _(primals, cotangent, output):
    x, y = primals
    return cotangent * y, cotangent * x      # one gradient per primal, in order
```

- The vjp receives `(primals, cotangents, outputs)`. **A single argument arrives as a bare array, not a 1-tuple.** In a one-input function, `primals[0]` is the first element, and the gradient comes out silently wrong. Two or more inputs arrive as a tuple. The same rule applies to outputs: one output gives a bare cotangent, several give a tuple.
- `.jvp(primals, tangents)` and `.vmap(inputs, axes) -> (outputs, out_axes)` register the other transforms, with the same bare-versus-tuple rule (a one-input vmap gets `axes == 0`, not `(0,)`).
- A missing transform falls back to differentiating the forward body, which does not reuse forward outputs. A fix for vmap and jvp of a custom function whose body is an opaque `mx.fast.metal_kernel` ships in 0.32.5. On 0.32.4, register `.vmap` and `.jvp` explicitly for such functions.
- Wrapping a `metal_kernel` belongs to `mlx-kernel`.

## Checkpointing

```python
block = mx.checkpoint(lambda x: mx.tanh(x @ w))           # wrong: w gets a ZERO gradient, no error
block = mx.checkpoint(lambda x, w: mx.tanh(x @ w))        # right: everything differentiated is an argument
layer_fn = nn.utils.checkpoint(layer)                      # module: passes its trainable parameters for you
```

`mx.checkpoint` differentiates only its arguments. A weight it closes over receives zeros (verified against the same function without `checkpoint`). `custom_function` has the same rule, because `checkpoint` is built on it. The forward activations inside the checkpointed function are dropped and recomputed during the backward pass. Checkpoint per transformer block, not per op: each boundary adds a recompute.

## export_function and import_function

```python
mx.export_function("f.mlxfn", fun, example_x)                  # one trace, fixed shapes
mx.export_function("g.mlxfn", fun, example_x, shapeless=True)  # same rules as shapeless compile
with mx.exporter("m.mlxfn", fun) as ex:                        # several signatures, one file
    ex(mx.ones(2)); ex(mx.ones(3))
f = mx.import_function("m.mlxfn")
(out,) = f(mx.ones(3))                                         # always returns a tuple
```

- Arrays the function closes over, such as a model's weights, are written into the file as constants. Exporting a model's forward exports its weights.
- Keyword arguments are supported (`export_function(path, fn, a=x)` then `f(a=x)`). Exporting the same signature twice through one `exporter` raises.
- A call that matches no exported signature raises `[import_function::call] No imported function found which matches the given positional and keyword arguments.` and lists the available signatures.
- `metadata="..."` stores a string with the export (since 0.32.1).
- Primitives without serialization raise `[export_function] Unable to serialize primitive <Name>`.
- Import validates trace input ids since 0.32.5.
