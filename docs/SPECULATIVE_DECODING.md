# Speculative Decoding

[README](../README.md)

DSpark proposes up to five future tokens with a small support model, and the
main model verifies them in one batched pass. An accepted prefix advances
generation by several tokens. It does not accelerate prefill, it is opt-in, and
poor acceptance can make it slower, so measure your own workload.

## Using it

DSpark is a separate support GGUF, not a standalone language model, and it must
match the main checkpoint. For Flash 0731:

```sh
./download_model.sh ds4f-q2
./download_model.sh ds4f-dspark
./ds4-server --dspark --mtp-model gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf
```

(`--mtp-model` is the flag that names the support file.) The support file adds
about 5.6 GiB of weights plus runtime state. DSpark runs for non-batched `ds4-server` requests; session-batched serving
([`--batched-session`](SERVER.md#multiple-sessions)) uses ordinary decoding.
The scheduler backs off when drafting is unproductive, and the defaults select
the fast paths, so no other flags or environment variables are needed.

On this branch every operation DSpark runs, including the verifier's 1-token
batches and the drafter's norms and projections, is TileLang-covered. On the
reference M5 Max, about three quarters of drafted tokens are accepted
(`DS4_DSPARK_STATS=1` prints the statistics at exit). A coding prompt
generates at about 50 tokens/s against 44 without DSpark; on prose the two are
about even.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy drafts
are accepted directly, and sampling resumes when the proposed suffix does not
match. This is deliberately more deterministic than ordinary sampling.
`--mtp-exact-sampling` preserves the ordinary target sampling distribution
instead: it accepts greedy proposals with their target probability and samples
from the remaining distribution on rejection.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. To compare against the target-only path, use
`--dspark-strict`, which disables the speculative acceptance path.
