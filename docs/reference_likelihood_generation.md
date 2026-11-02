# Reference-policy likelihood generation

This stage generates fresh candidates through the existing OpenAI-compatible
generator and records the likelihood assigned by that same generator. It does
not run a reward model or select Best-of-N candidates.

## Server requirement

The chat-completions server must support `logprobs=true` and return chosen-token
logprobs in `choices[0].logprobs.content`. The configured model alias must point
to the intended reference-policy checkpoint.

## Smoke run

Start the server configured in
`configs/llama32_3b_generation_with_logprobs.json`, then run:

```bash
python -m ensemble.generation.generate_reference_candidates \
  --dataset gsm8k \
  --split test \
  --trial-id 1 \
  --inference-config configs/llama32_3b_generation_with_logprobs.json \
  --output-dir outputs/reference_generation/gsm8k_smoke \
  --num-samples 2 \
  --limit-problems 2
```

For the full 512-candidate run, remove `--limit-problems` and set
`--num-samples 512`.

## Outputs

- `responses/<candidate_id>.json`: resumable raw API response, including the
  complete server-provided logprob object.
- `generated_candidates.jsonl`: deterministically ordered experiment records
  with prompt,
  response, ground truth, correctness, finish reason, generation settings,
  Sequence NLL, Average NLL, and token count.

Chosen-token logprobs remain in each raw per-candidate response. They are not
duplicated in the canonical JSONL, which keeps full 512-candidate runs smaller.

The primary likelihood fields are:

```text
sequence_logprob = sum_t log p(y_t | x, y_<t)
sequence_nll     = -sequence_logprob
average_nll      = sequence_nll / token_count
```

The API reports generated content-token logprobs. Prompt tokens and an
unobserved EOS token are not included. `sequence_probability` is also stored,
but can underflow to zero for long responses; use `sequence_logprob` or
`sequence_nll` for analysis.

Generation is split into request chunks, and canonical records are written as a
stream to limit host memory. Re-running the same command reuses completed
per-candidate raw responses and reconstructs the canonical JSONL file.

If `--sample-seed-base` is provided, each request receives a deterministic,
distinct seed derived from the problem index and sample index. A `seed` in the
JSON generation config is treated as this base automatically; it is not reused
unchanged for all 512 requests.
