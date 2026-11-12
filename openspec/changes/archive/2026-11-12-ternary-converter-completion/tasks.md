# Tasks

## 1. Correctness fixes (biases, conv weights, sharding, dtypes)

- [x] 1.1 Fold original biases into per-channel scales in `Layer.from_dense/3` (or store as FP16 record) and verify a converted model's bias list is non-zero when source biases are non-zero
- [x] 1.2 Update `Architectures` to stop silently skipping biases; route them to the FP16 record path and verify `info` reports them
- [x] 1.3 Add 4D conv reshape in `Architectures.transform/3` and verify a 4D conv tensor converts to a 2D ternary layer without crashing
- [x] 1.4 Implement `Safetensors.read_sharded/1` (index JSON + relative shard paths) and verify a two-shard fixture loads all tensors
- [x] 1.5 Make `Safetensors.decode/1` return `{:error, _}` on unknown dtype instead of raising, and verify with a test
- [x] 1.6 Add `converter_version` to export metadata and verify it is present in loaded metadata
- [x] 1.7 Add tests for the above in `architectures_test.exs`, `safetensors_test.exs`, `pipeline_test.exs`

## 2. Sparse packing and stats

- [x] 2.1 Add `:packing` (`:auto`/`:dense`/`:sparse`) and `:sparse_threshold` options to `Layer.from_dense/3` and verify auto-selection at the threshold
- [x] 2.2 Store the chosen packing mode in a new `Layer.packing` field and serialize/deserialize it in `Layer.to_binary/from_binary`
- [x] 2.3 Update `Exporter` and the round-trip validator to dispatch on `packing` and verify round-trip for both modes
- [x] 2.4 Update `Stats.compute/1` to account for sparse payloads and verify compression numbers
- [x] 2.5 Document the 15-bit sparse index limit and fall back to dense with a warning above it, verified by a test
- [x] 2.6 Add tests in `packer_test.exs` and `layer_test.exs`

## 3. Calibration CLI and mixed precision

- [x] 3.1 Add `--sparsity-target`, `--clip`, and `--calibration-data` flags to the real-model conversion path and verify each flag changes output
- [x] 3.2 Wire `--clip` to `Quantizer.compute_scales_clipped/3` and verify scales differ from unclipped
- [x] 3.3 Wire `--sparsity-target` to `Quantizer.auto_threshold/2` on real models and verify reported sparsity is within tolerance
- [x] 3.4 Add `:target_similarity` to `Pipeline.run/3` calling `Sensitivity.auto_mixed_precision/4` and verify promoted layers become FP16
- [x] 3.5 Add tests for `Sensitivity` module (analyze, auto_mixed_precision, edge cases) in a new `sensitivity_test.exs`
- [x] 3.6 Reconcile `details/13` Quantizer signatures with implementation (`quality_metrics/2`, bisection `auto_threshold`, plain `def`)

## 4. QAT path

- [x] 4.1 Implement `TernaryConverter.QATLayer` struct with shadow weights, scales, bias, delta and `from_pretrained/3`
- [x] 4.2 Implement `ternarize/1`, `forward/2`, and STE gradient flow and verify gradients reach shadow weights in a test
- [x] 4.3 Implement `update/3` (SGD) and a training loop with loss, optional distillation, sparsity reg, and gradual delta ramp
- [x] 4.4 Implement `freeze/1` producing a standard `Layer` and verify the frozen `.tbin` loads and round-trips
- [x] 4.5 Add a `qat` CLI command and tests for the QAT module

## 5. Inference runtime

- [x] 5.1 Implement `TernaryConverter.Runtime` that loads a `.tbin` plus a config and chains layers via `forward_nx`
- [x] 5.2 Implement multi-head attention using ternary Q/K/V/O projections and scaled dot-product attention
- [x] 5.3 Implement LayerNorm/RMSNorm and softmax in FP32 and verify softmax rows sum to 1
- [x] 5.4 Implement KV cache with prefill (full sequence) and decode (single token) and verify cached decode matches full recompute
- [x] 5.5 Add `mix infer` command and a determinism test (identical inputs → identical outputs)

## 6. Mix tasks, escript, and validation harness

- [x] 6.1 Add `lib/mix/tasks/convert.ex`, `infer.ex`, `analyze.ex` implementing the documented flags
- [x] 6.2 Configure and verify `mix escript.build` produces a runnable binary
- [x] 6.3 Make CLI `validate` invoke the full harness (round-trip, sparsity, cosine similarity when reference given) and verify output
- [x] 6.4 Add perplexity computation when a reference model/tokenizer/dataset is provided, else document the manual procedure
- [x] 6.5 Update `details/13` path references from `examples/` to `tools/` and remove stale snippets

## 7. Integration and gate measurement

- [x] 7.1 End-to-end test: convert a small BERT checkpoint, run runtime, report cosine similarity and (if available) perplexity
- [x] 7.2 Record Phase 1→2 gate result (PTQ perplexity delta ≤ 2%) in the change notes and update roadmap if missed
- [x] 7.3 Run `mix test` and `openspec validate` and verify all pass