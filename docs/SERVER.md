# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

Pass `-m FILE` for an explicit model. The Metal kernels and the tuned config
are embedded in the binary, so the server needs nothing but the GGUF.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |

The model names accepted by the endpoints are aliases for the one GGUF passed at startup.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. `reasoning_effort=max` selects Think
Max only with sufficient context; otherwise it falls back to normal thinking.
`xhigh` maps to normal thinking, not Think Max. Use `think:false`, a disabled
thinking object, or a non-thinking model alias for direct answers.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
preallocates independent KV states and queues requests when all slots are busy.
Choose context and slot count together: a context that fits once may not fit
four times. Idle slots can be cached before reuse; active requests are not evicted.

Concurrent decode rows are batched where the shared-expert and attention
projections allow it, which gives an aggregate speedup; the batched kernels sum
in a different order than one-session decoding, so a greedy continuation can
differ from the one-at-a-time result at a near-tied token. Long prefills yield to
active decoders in bounded intervals, normally 128 tokens;
`--mixed-prefill-quantum N` changes that interval for testing. Session-batched
serving uses ordinary target decoding, without DSpark.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Their defaults are in `ds4_server.c`.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace /tmp/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../ds4_kvstore.h) and
[ds4_kvstore.c](../ds4_kvstore.c); payload handling is in [ds4.c](../ds4.c).
