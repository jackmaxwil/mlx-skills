---
name: mlx-serve
description: Serving and generating with MLX models — choosing between generate, stream_generate, generate_step, batch_generate and BatchGenerator, the decode loop and chunked prefill, prompt caches and what they hold after generation, reuse, trim, save and load, cache sizing, KV-cache quantization and its costs, speculative decoding constraints, samplers and logits processors, chat templates, double BOS, thinking defaults and EOS sets, continuous batching, mlx_lm.server flags, batchability and security. Load when writing a generation loop, an inference service or a chat application on mlx-lm.
---

# Generation and serving

Written against **mlx 0.32.4** and **mlx-lm 0.32.2**. Why the decode loop uses `async_eval` belongs to `mlx`. The wired limit that generation sets and memory budgeting belong to `mlx-memory`. Generating from worker threads belongs to `mlx-threads`. Multi-machine serving with `--pipeline` and `sharded_load` belongs to `mlx-distributed`.

## Which entry point

| Need | Use | Returns |
| --- | --- | --- |
| One string, blocking | `generate(model, tok, prompt, max_tokens=..., verbose=False, **kw)` | `str` |
| Streamed text with stats | `stream_generate(model, tok, prompt, max_tokens=..., draft_model=None, **kw)` | `GenerationResponse` with `text`, `token`, `logprobs`, `from_draft`, `prompt_tokens`, `prompt_tps`, `generation_tokens`, `generation_tps`, `peak_memory`, `finish_reason` |
| Token ids, your own detokenizing or stopping | `generate_step(prompt_array, model, **kw)` | `(token_int, logprobs)` pairs |
| A fixed list of prompts at once | `batch_generate(model, tok, prompts, max_tokens=..., prompt_caches=None, return_prompt_caches=False)` | `BatchResponse` with `.texts` (and `.caches`) |
| Requests arriving over time | `BatchGenerator(model, max_tokens=..., stop_tokens=..., sampler=..., completion_batch_size=32, prefill_batch_size=8)` | continuous batching |
| An OpenAI-compatible HTTP API | `mlx_lm.server` | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` |

Keyword arguments flow from `generate` to `stream_generate` to `generate_step`: `sampler`, `logits_processors`, `max_kv_size`, `prompt_cache`, `prefill_step_size` (2048), `kv_bits`, `kv_group_size` (64), `quantized_kv_start` (5000), `prompt_progress_callback`, and `num_draft_tokens` when there is a draft model.

`stream_generate` stops on any id in `tokenizer.eos_token_ids`. The final response has `finish_reason` set to `"stop"` or `"length"`. Earlier responses have `None`.

## Writing your own loop

Copy the shape of `generate_step` rather than inventing one:

1. **Prefill in chunks.** Run `model(prompt[:, :step], cache=cache)` for chunks of `prefill_step_size`, call `mx.eval([c.state for c in cache])` after each, then `mx.clear_cache()`. Keep the **last** prompt token out of the prefill and feed it through the first decode step, which produces the first logits.
2. **Decode one ahead.** Build step n+1 and `mx.async_eval` it before calling `.item()` on step n.
3. Run the whole loop inside `with mx.stream(generation_stream):` (a thread-local stream in `mlx_lm.generate`).
4. Wrap the whole generation in `with wired_limit(model, [generation_stream]):` from `mlx_lm.generate`. `stream_generate` does this for you, and the bare `generate_step` does not.

```python
logits = model(y[None], cache=cache)[:, -1, :]
logprobs = logits - mx.logsumexp(logits, keepdims=True)
y = sampler(logprobs)              # sampler receives log-probabilities, not raw logits
```

Prefill activations and the attention-score block both grow with `prefill_step_size`. When long prompts run out of memory, lower it first.

## Prompt caches

```python
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.models.cache import save_prompt_cache, load_prompt_cache
cache = make_prompt_cache(model, max_kv_size=None)
```

- **Generation mutates the cache in place.** After `generate_step(prompt, ..., max_tokens=N, prompt_cache=c)` runs to completion, `c[0].offset == len(prompt) + N`. Every yielded token, including the last one, has been fed back in. On the next turn, feed only the tokens after that point, or `trim_prompt_cache(c, k)` to drop the last `k`.
- **Shared prefix.** Prefill the system prompt once, then give each request `copy.deepcopy(base_cache)`. Passing the same cache to two requests corrupts both. mlx-lm's `LRUPromptCache` deep-copies on fetch for this reason.
- **Trimming is not always possible.** `RotatingKVCache` is trimmable only while `offset < max_size`. `trim_prompt_cache` returns `0` and does nothing when any layer cannot trim, so check the return value or `can_trim_prompt_cache` first.
- **Files.** `save_prompt_cache(path, cache, {"model": repo, ...})` writes safetensors with string metadata. `load_prompt_cache(path, return_metadata=True)` rebuilds the cache classes by name. CLI: `mlx_lm.cache_prompt --model M --prompt-cache-file p.safetensors --prompt "..."`, then `mlx_lm.generate --prompt-cache-file p.safetensors --prompt "next"`. The generate CLI refuses a different `--model` than the one recorded, and refuses mismatched `--kv-bits` and `--kv-group-size`.

### max_kv_size is silently ignored by many models

`make_prompt_cache(model, max_kv_size=n)` builds `RotatingKVCache(max_size=n, keep=4)` **only when the model has no `make_cache` method**. 53 model files in mlx-lm 0.32.2 define one, `llama` included, and they get their own caches with no size bound. `--max-kv-size` is therefore a no-op for them. Check what you got:

```python
cache = make_prompt_cache(model, max_kv_size=4096)
assert all(type(c).__name__ == "RotatingKVCache" for c in cache), [type(c).__name__ for c in cache]
```

For a bounded context on such a model, build the list yourself (`[RotatingKVCache(max_size=n, keep=4) for _ in model.layers]`) for full-attention layers only. Linear and SSM layers keep their `ArraysCache`. With a rotating cache, tokens past the window are forgotten, except the first `keep`.

## KV-cache quantization

`kv_bits=4|8` converts each `KVCache` to a `QuantizedKVCache` once its offset reaches `quantized_kv_start` (default 5000), with `kv_group_size` 64.

- Attention over a quantized cache is not fused. It materializes a `prefill_step_size × context` score matrix, which can be larger than the memory the quantization saved. Lower `prefill_step_size` (for example 512) together with `kv_bits`.
- Sliding-window models crash with `NotImplementedError: RotatingKVCache Quantization NYI` once the threshold is crossed.
- Models with attention sinks (gpt-oss) raise `Quantized SDPA does not support attention sinks.`
- The server serves one request at a time when `--kv-bits` is set.

## Speculative decoding

`stream_generate(model, tok, prompt, draft_model=draft, num_draft_tokens=3)` gives the same tokens as the target model alone under greedy sampling. Each response's `from_draft` says whether the draft proposed that token.

- The draft must share the target's tokenizer and vocabulary.
- Both caches must be trimmable, because rejected drafts are rolled back with `trim_prompt_cache`. Rotating (sliding-window) caches past their window break this.
- `max_kv_size` does nothing with a draft model: `stream_generate` silently drops it, and calling `speculative_generate_step` directly raises `TypeError: speculative_generate_step() got an unexpected keyword argument 'max_kv_size'`.
- Measure `generation_tps` with and without the draft. Every rejected draft token is wasted work, so a draft that rarely agrees makes generation slower.
- In mlx-lm 0.32.2, logits processors (repetition penalties) in speculative mode do not see the prompt tokens in their history. That was fixed after 0.32.2.

## Sampling

```python
from mlx_lm.sample_utils import make_sampler, make_logits_processors
sampler = make_sampler(temp=0.7, top_p=0.9, min_p=0.0, top_k=0, xtc_probability=0.0, xtc_threshold=0.1)
procs = make_logits_processors(logit_bias=None, repetition_penalty=1.1, repetition_context_size=20,
                               presence_penalty=None, frequency_penalty=None)
```

- `temp=0` (the default) is argmax and ignores every other sampler parameter. Penalties apply as logits processors, before the sampler.
- The compiled samplers capture `mx.random.state`. Seed with `mx.random.seed(s)` for reproducible output. Per-thread RNG belongs to `mlx-threads`.
- A sampler is `f(logprobs) -> token`, and a logits processor is `f(tokens_so_far, logits) -> logits`. Both run on the generation stream, so they must stay as MLX ops: a NumPy round trip inside them forces a sync every token.

## Chat templates and tokenizers

Always use the `TokenizerWrapper` that `mlx_lm.load` returns.

```python
# Wrong: a raw transformers 5 tokenizer returns a BatchEncoding dict.
mx.array(hf_tok.apply_chat_template(msgs, add_generation_prompt=True))
# ValueError: Invalid type transformers.tokenization_utils_base.BatchEncoding received in array initialization.

# Wrong: the template already contains BOS, and encode() adds a second one.
ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))

# Right: token ids straight from the wrapper.
ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
```

If you must template text and encode it later, use `tok.encode(text, add_special_tokens=False)`. `stream_generate` given a **string** adds BOS only when the string does not already start with it.

- **Thinking.** The wrapper passes `enable_thinking=tok.has_thinking` unless you set it, so reasoning models think by default. Turn it off per call (`enable_thinking=False`), on the server with `--chat-template-args '{"enable_thinking": false}'`, or per request with `chat_template_kwargs`. The wrapper renames the kwarg for templates that call it something else.
- **EOS.** `tok.eos_token_ids` is a set, merged from the tokenizer and `config.json`'s `eos_token_id`. Chat models often end turns with a token that is not the tokenizer's `eos_token` (`<|im_end|>`, `<|eot_id|>`). Add one with `tok.add_eos_token("<|im_end|>")`, `--extra-eos-token` in `mlx_lm.generate`, or `--eos-token` in `mlx_lm.cache_prompt`.
- A tokenizer without a template: `generate` has `--ignore-chat-template`, and `--use-default-chat-template` sets one (`mlx_lm.server` also takes `--chat-template`). Do not wrap base-model prompts in a chat template.

## Continuous batching

```python
gen = BatchGenerator(model, max_tokens=256, stop_tokens=[[eos] for eos in tok.eos_token_ids])
uids = gen.insert([ids_a, ids_b], max_tokens=[128, 512])
while active:
    for r in gen.next_generated():          # r.uid, r.token, r.logprobs, r.finish_reason
        ...
gen.remove([uid], return_prompt_caches=True)
gen.close()                                 # restores the wired limit
```

- Prompts are left-padded into `BatchKVCache`, which needs every cache type in the model to implement `merge`. A model is batchable only when all of `make_prompt_cache(model)` have it. Hybrid and SSM caches do, and quantized caches do not.
- `BatchGenerator` sets the wired limit in `__init__`. Call `close()` (or rely on `__del__`) when done.
- `batch_generate(..., return_prompt_caches=True)` returns per-prompt caches that can be passed back as `prompt_caches=` for the next turn of each conversation.

## mlx_lm.server

```bash
mlx_lm.server --model M --host 127.0.0.1 --port 8080 --max-tokens 512 \
  --decode-concurrency 32 --prompt-concurrency 8 --prefill-step-size 2048 \
  --prompt-cache-size 10 --prompt-cache-bytes 8GB --chat-template-args '{"enable_thinking":false}'
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--decode-concurrency` / `--prompt-concurrency` | 32 / 8 | batch sizes when requests are batchable |
| `--prompt-cache-size` / `--prompt-cache-bytes` | 10 / no byte limit | LRU of KV caches for prefix reuse across requests |
| `--draft-model`, `--num-draft-tokens` | none, 3 | speculative decoding; **disables batching** |
| `--kv-bits`, `--kv-group-size`, `--quantized-kv-start` | none, 64, 5000 | KV quantization; **serves one request at a time** |
| `--temp`, `--top-p`, `--top-k`, `--min-p`, `--max-tokens` | 0.0, 1.0, 0, 0.0, 512 | defaults that requests can override |
| `--allowed-origins` | `*` | CORS |
| `--pipeline` | off | pipeline instead of tensor parallel under `mlx.launch` |

- **A request's `model` field makes the server load that path or Hugging Face repo.** Upstream marks the server as not for production, with only basic security checks. Keep the default `--host 127.0.0.1`. Never expose it directly; put a proxy in front that allowlists the `model` value and strips `adapters` and `draft_model`.
- The prompt cache is matched on token prefixes, so a stable system prompt and stable tool definitions at the start of `messages` are what make reuse hit. A timestamp in the system prompt defeats it.
