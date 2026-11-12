# Design

## Context

The converter is an Elixir application (`tools/balanced_ternary_engine`) with a generic `Pipeline` that reads safetensors, classifies tensors via `Architectures` presets, quantizes, packs, and exports `.tbin`. Existing presets cover `llama`, `gpt2`, `bert`, `vit`, `generic`. The three requested tools need presets/recipes plus CLI entry points and validation hooks. See `proposal.md` for motivation.

## Goals / Non-Goals

**Goals:**
- Reuse the existing pipeline for all three families; add presets and thin converter wrappers.
- Provide CLI commands/profiles `yolo`, `imagenet`, `llm`.
- Keep changes additive; no `.tbin` format changes.

**Non-Goals:**
- Implementing full model inference runtimes for YOLO/ImageNet (only conversion + basic validation).
- Training/fine-tuning converters (QAT remains separate).
- Supporting models >= 8B for the LLM tool.

## Decisions

- **Decision: Reuse `Pipeline.run/3` with new architecture presets** instead of separate pipelines.
  - Rationale: minimizes duplication; classification/calibration already configurable.
  - Alternative: standalone converter modules per family — rejected as duplicated logic.

- **Decision: Add `:yolo`, `:imagenet`, `:llm` presets to `Architectures`.**
  - YOLO: ternarize conv/linear backbone weights; keep detection head, norms, and biases FP16.
  - ImageNet: ternarize backbone; keep classifier head and norms FP16.
  - LLM: reuse `:llama`-style policy but cap at <8B and require sharded input path.

- **Decision: CLI uses `--profile` selector (yolo|imagenet|llm) mapping to arch + defaults.**
  - Rationale: avoids proliferating commands; consistent with existing `--arch`.

- **Decision: Validation hooks report family-appropriate metrics but reuse round-trip + sparsity.**
  - Detection/classification heads remain FP16; cosine similarity when reference provided.

## Risks / Trade-offs

- [Missing family-specific tensor naming] → fall back to `:generic` heuristics and warn.
- [Large LLM memory] → require sharded input and stream tensors; reject >8B by parameter count.
- [Conv head misclassification] → conservative: keep any tensor matching head/classifier/detect as FP16.

## Migration Plan

Additive only. Existing commands unchanged. New profiles optional. Rollback by reverting new presets/CLI flags.

## Open Questions

None.