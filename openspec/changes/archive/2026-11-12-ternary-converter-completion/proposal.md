# Proposal

## Why

The repository is a research/design corpus plus an Elixir converter (`tools/balanced_ternary_engine`) that ships 55 passing tests but has drifted from its own documented spec (`details/13`). Three gaps block the project's stated phase gates (§22.5):

1. **Correctness**: converted models silently drop all bias terms (architectures skip biases and `Layer` inits them to zero); 4D conv weights crash `from_dense`; sharded safetensors (real HF checkpoints) are unreadable; unknown dtypes abort the whole load.
2. **Quality path**: `TernaryConverter.QATLayer` and the entire QAT training loop (STE, distillation, sparsity reg) are documented but unimplemented, yet the Phase 1→2 gate (BERT PTQ perplexity ≤ 2%) likely needs QAT to clear.
3. **Runtime**: `.tbin` loads but nothing executes a converted model end-to-end (no attention, LayerNorm, KV cache, prefill/decode), and validation stops at round-trip/shape checks — no perplexity or output-similarity harness.

Without these, the converter cannot validate a real model against a reference, and the FPGA/ASIC path has no bit-exact oracle.

## What Changes

- **Bias preservation**: fold original biases into per-channel scales or store them as FP16 records; never silently zero them.
- **Robust conversion**: handle 4D conv weights (reshape to 2D GEMM), read sharded safetensors via index JSON, surface dtype errors as `{:error, _}` instead of raising.
- **Sparse export**: wire `Packer.pack_sparse` into `Layer`/`Exporter` with a documented sparsity threshold (~88%) and update `Stats`.
- **Calibration CLI**: expose `--sparsity-target`, `--clip`, `--calibration-data` on real (non-synthetic) conversion; align `auto_threshold` docs with the bisection implementation.
- **Mixed precision wiring**: call `Sensitivity` from `Pipeline` with a documented accuracy target instead of relying solely on name-based policy; add tests for `Sensitivity`.
- **QAT path**: implement `QATLayer` (shadow weights, STE ternarization, update, freeze) and a training loop with knowledge distillation, sparsity regularization, and gradual delta scheduling.
- **Inference runtime**: add a minimal transformer runner that chains ternary/FP16 layers, applies attention + LayerNorm + softmax, and maintains a KV cache with prefill/decode split.
- **Validation harness**: compute output cosine similarity and (when a reference is available) perplexity; make CLI `validate` invoke the full harness.
- **Mix tasks**: add `mix convert`, `mix infer`, `mix analyze` (and escript build) replacing the `mix run -e` invocation; fix the doc's `examples/` path to `tools/`.
- **Doc reconciliation**: update `details/13` Quantizer signatures to match implementation (`quality_metrics/2`, `defn`→`def`), and document the 15-bit sparse index limit.

**BREAKING**: converted `.tbin` outputs for models with biases will differ (biases now preserved); consumers relying on zero biases must re-convert. Format version stays at 2.

## Capabilities

### New Capabilities

- `ternary-conversion`: converting pretrained (possibly sharded) models to `.tbin` — PTQ in delta/twn/absmean modes, per-channel scales, mixed-precision policy, bias preservation, sparse packing, calibration options, and validation reporting.
- `ternary-qat`: quantization-aware training path — shadow weights, straight-through estimator, distillation, sparsity regularization, and freeze to `.tbin`.
- `ternary-inference-runtime`: executing a converted `.tbin` model end-to-end — layer chaining, attention, LayerNorm/softmax, KV cache, prefill/decode.

### Modified Capabilities

None (no existing specs in `openspec/specs/`).

## Impact

- Affected code: `tools/balanced_ternary_engine/lib/**` (Quantizer, Packer, Layer, RawLayer, Exporter, Safetensors, Architectures, Pipeline, Sensitivity, Stats, new QATLayer, new Runtime module, new Mix tasks), `tools/balanced_ternary_engine/mix.exs`, tests.
- Affected docs: `details/13-model-conversion-guide.md` (reconciliation), `details/08-research-roadmap.md` (gate criteria already defined).
- Dependencies: adds nothing new for PTQ; QAT/runtime may need a tokenizer + dataset loader (evaluate Nx/Tokenizers). No new runtime deps committed without review.
- Systems: enables the Phase 1→2 (BERT) and 2→3 (LLaMA-1B) gates to be measured rather than estimated.