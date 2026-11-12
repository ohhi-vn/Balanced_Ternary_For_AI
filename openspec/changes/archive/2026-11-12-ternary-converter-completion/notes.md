# Implementation Notes

## Phase 1→2 gate (task 7.2)

The PTQ perplexity delta ≤ 2% gate requires a real BERT checkpoint and a
perplexity harness with a tokenizer + dataset. Neither is available in this
repository (Elixir-only, no PyTorch/Python sidecar). The converter and
runtime now support the measurement path; the recorded gate result is
deferred until a checkpoint is supplied via `mix convert` + `mix infer`.

Converter version: 0.1.0. Tests: 73 passing. openspec validate: pass.