# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Flash checkpoints are not interchangeable for this test. `flash-0731/` holds the
July 31 checkpoint, the one this branch targets and the default used by
`ds4_test`; a GGUF whose name contains `0731` must use it. A mismatch between
checkpoint and fixture is an invalid test, not a model-quality result.

Files in `flash-0731/`:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON. It is
  intentionally trivial to parse from C: each case points to a prompt file and
  each expected token is hex-encoded by bytes. The JSON files stay in the tree so
  the fixture can be audited against the API response.
- `local-golden.vec`: local top-k/logit fixture captured from the matching GGUF.
  It catches substantial backend drift that can keep the same greedy token while
  damaging the logits distribution.

Run the checks:

```sh
./ds4_test --logprob-vectors
./ds4_test --local-golden-vectors
```

Regenerate the official vectors (the checkpoint label is required; confirm which
checkpoint the API is serving before choosing it, and put a future checkpoint
into a new directory instead of replacing this one):

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py \
  --checkpoint 0731
```

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/flash-0731/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```
