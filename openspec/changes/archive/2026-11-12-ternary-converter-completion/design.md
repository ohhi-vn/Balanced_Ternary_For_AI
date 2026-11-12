# Design

## Context

The converter lives in `tools/balanced_ternary_engine/` (Elixir, Nx, Jason). It already implements PTQ in three modes, 10-trits-per-16-bit packing, sparse packing primitives, a v2 `.tbin` format mixing ternary and FP16 records, architecture policies for llama/gpt2/bert/vit/generic, and a Sensitivity module that is unused. The gap inventory is in the exploration that produced this change (Thread 1) and the phase gates are in `details/22` §22.5.

No existing specs exist (`openspec/specs/` is empty), so every capability here is new. The runtime is Elixir-only; there is no PyTorch in the repo, so perplexity validation will rely on either an optional Python sidecar or a documented manual procedure.

## Goals / Non-Goals

**Goals:**
- Make conversion correct on real HF checkpoints (biases, conv weights, sharding, dtypes).
- Add the QAT path and a minimal transformer runtime so the Phase 1→2 and 2→3 gates become measurable.
- Wire existing but unused pieces (sparse packing, sensitivity, clipping, auto-threshold) into the pipeline.
- Provide Mix tasks + escript and a full validation harness.

**Non-Goals:**
- ASIC/FPGA RTL (Phase 4–5) — out of scope; this change stops at a software oracle and a measured gate.
- Replacing the Elixir toolchain with Python/PyTorch.
- Changing the `.tbin` format version (stays at 2; bias storage already exists in the Layer struct).
- Training a production model from scratch (only QAT fine-tuning of a pretrained checkpoint).

## Decisions

### D1: Bias handling — fold into scales, keep FP16 fallback
- **Choice**: Fold biases into per-channel scales where the layer is ternary (absorb bias into the scale/zero-point), and store biases that cannot be folded as an FP16 record alongside the layer. Never zero them.
- **Rationale**: Folding preserves the (w·x + b) semantics with zero storage cost for ternary layers; FP16 fallback covers norms/biases that are quantization-sensitive.
- **Alternative**: Store every bias as a separate FP16 vector — simpler but bloats `.tbin` and diverges from the documented "sensitive tensors stay FP16" policy.

### D2: Sharded input — optional index JSON, single-file default
- **Choice**: `Safetensors.read/1` keeps single-file behavior; add `Safetensors.read_sharded/1` taking a path to `model.safetensors.index.json`, resolving shard paths relative to the index file. Pipeline auto-detects sharding when `--input` points at a directory or index file.
- **Rationale**: Backward compatible; no change to the common single-file path.
- **Alternative**: Always require an index file — breaks the existing CLI and tests.

### D3: 4D conv weights — reshape to 2D at classify time
- **Choice**: In `Architectures.transform/3`, reshape `{out, in, kh, kw}` conv weights to `{out, in*kh*kw}` before quantization, recording the original shape in layer metadata for un-flattening by the runtime.
- **Rationale**: Keeps the ternary core (2D `{out, in}`) unchanged; reshape is lossless and reversible.
- **Alternative**: Generalize `Layer` to N-D tensors — large blast radius, not justified.

### D4: QAT — shadow weights in FP32, STE via identity gradient
- **Choice**: `QATLayer` holds `weight_shadow` (FP32). Forward ternarizes with the configured delta and scales. Backward uses Nx's gradient through a custom STE (gradient of identity for the ternarize op, i.e., `w_ste = w_shadow + (w_q - w_shadow).detach()` semantics approximated by adding a zero-gradient copy). Update via SGD/Adam on shadow weights. Freeze re-quantizes shadow weights into a standard `Layer`.
- **Rationale**: Matches the documented `QATLayer` shape; keeps the inference format unchanged.
- **Alternative**: Learnable scale factors (TTQ-style) — deferred; scales stay MSE-optimal recomputed per epoch.

### D5: Runtime — minimal transformer, no batching beyond batch=1 for v1
- **Choice**: A `Runtime` module that loads a `.tbin` plus a small config (hidden size, heads, layers, vocab) and runs prefill (full sequence) and decode (single token) with a KV cache. Attention uses the ternary projections' `forward_nx`. Activations stay FP32; softmax/LayerNorm in FP32.
- **Rationale**: Smallest unit that can produce perplexity, which is the gate metric. Batching and INT8 activations deferred.
- **Alternative**: Full Nx/Bumblebee integration — too heavy; revisit after gates are measured.

### D6: Validation — perplexity via optional reference, cosine similarity always
- **Choice**: `validate` always computes round-trip, sparsity, and output cosine similarity against a user-supplied reference activation (or synthetic). Perplexity is computed only when a tokenizer + dataset + reference model are provided via `--calibration-data`/`--reference`; otherwise a documented manual procedure (run `mix infer` over a corpus) is recorded. No Python dependency added to the core app.
- **Rationale**: Keeps the Elixir app dependency-light while still enabling the gate; perplexity is a measurement, not a converter feature.
- **Alternative**: Bundle a Python sidecar — rejected for v1 to avoid a second runtime.

### D7: Sensitivity — wire into Pipeline behind `--target-similarity`
- **Choice**: Add an optional `:target_similarity` option to `Pipeline.run/3`. When set, call `Sensitivity.auto_mixed_precision/4` over the ternary candidates (using a small sample input derived from the first layer's shape) and reclassify promoted layers as FP16. Default (unset) keeps the current name-based policy.
- **Rationale**: Opt-in; existing tests and default behavior unchanged.
- **Alternative**: Always run sensitivity — too slow and requires a representative input that may not exist.

### D8: Sparse packing — threshold-gated in Layer.from_dense
- **Choice**: `Layer.from_dense/3` accepts `:packing` (`:auto` default, `:dense`, `:sparse`) and `:sparse_threshold` (default 0.88). When `:auto`, compute sparsity and switch. Store the chosen mode in the layer (new `packing` field). Update `Stats` and the round-trip validator to dispatch on `packing`.
- **Rationale**: Keeps the decision local to the layer; stats and validation follow.
- **Alternative**: Decide in Exporter — too late; packing affects the payload bytes.

### D9: Mix tasks over escript-only
- **Choice**: Add `lib/mix/tasks/convert.ex`, `infer.ex`, `analyze.ex` implementing the documented flags, and keep the escript entry point. Fix the doc path from `examples/` to `tools/`.
- **Rationale**: Matches the documented UX; Mix tasks integrate with the existing `mix` workflow.

### D10: Doc reconciliation
- **Choice**: Update `details/13` to match implementation: `quality_metrics/2`, plain `def` (not `defn`), bisection `auto_threshold`, 15-bit sparse index limit, and remove the stale `Calibrator`/`Mix.Tasks.Convert` snippets that don't match (or implement them per D9).

## Risks / Trade-offs

- **Bias folding changes outputs** → re-conversion required for any existing `.tbin`; bump documented as breaking in proposal. Mitigation: keep format v2, add a `converter_version` metadata field.
- **QAT needs a loss/optimizer** → Nx has no built-in optimizer in this dep set; implement a minimal SGD/Adam in ~100 LOC. Mitigation: scope QAT to SGD first, Adam as a follow-up task.
- **Perplexity without Python** → may be infeasible to compute on large corpora in pure Elixir. Mitigation: document the manual procedure and make the harness optional; the gate can be measured with a small corpus.
- **Sensitivity promotion needs a sample input** → derive from first layer's shape (random normal). Mitigation: only run when `--target-similarity` is passed; warn if input is synthetic.
- **Runtime scope creep** → attention + KV cache is substantial. Mitigation: gate the runtime behind its own spec; PTQ/correctness tasks can ship independently.
- **Sparse packing index limit** (32767 trits) → large layers cannot use sparse encoding. Mitigation: document limit; for oversized layers fall back to dense with a warning.

## Migration Plan

1. Ship correctness fixes (biases, conv, sharding, dtypes) + tests. No format change.
2. Add Mix tasks + escript build; update docs.
3. Wire sparse packing + sensitivity + calibration CLI (opt-in flags).
4. Implement QAT module + training loop behind a new `qat` command.
5. Implement runtime behind a new `infer` command; add perplexity harness.
6. Run Phase 1→2 gate measurement on a BERT checkpoint; record results.

Rollback: each step is additive behind flags; revert the commit range if a gate regresses.

## Open Questions

- Should QAT support Adam in v1, or is SGD sufficient to clear the Phase 1→2 gate? (Recommend: SGD first; measure; escalate if gate missed.)
- Is a small (≤10k token) corpus acceptable for the perplexity gate, or does the gate require a full WikiText run? (Recommend: small corpus for iteration; full run for the recorded gate.)