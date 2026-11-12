# Proposal

## Why

The converter currently exposes generic architecture policies (`llama`, `gpt2`, `bert`, `vit`, `generic`) but has no first-class, repeatable path for three of the most common deployment targets: YOLO object detectors, ImageNet image classifiers, and decoder-only LLMs under 8B parameters. Users must hand-tune architecture selection, calibration, and validation, which yields inconsistent results and no documented acceptance criteria. Dedicated converter tools give each family a tested recipe with sensible mixed-precision and calibration defaults.

## What Changes

- Add a YOLO converter tool that handles detection backbones and heads (convolution weights reshaped to GEMM form), keeps detection heads and normalization at FP16, and reports detection-oriented validation (mAP-style sanity checks).
- Add an ImageNet converter tool for classification backbones (ResNet/MobileNet/EfficientNet/ViT-style), keeping the classifier head and normalization at FP16, with top-1/top-5 oriented validation.
- Add an LLM (< 8B) converter tool for sharded decoder-only checkpoints, with memory-aware chunking, decoder mixed-precision policy, and perplexity/cosine-similarity validation.
- Expose the three tools through the CLI (dedicated commands or a `--profile` selector) and document their usage.
- Add architecture presets and tests covering each family.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `ternary-conversion`: add requirements for YOLO, ImageNet, and sub-8B LLM converter tools (architecture policy, calibration defaults, validation reporting).

## Impact

- Affected code: `tools/balanced_ternary_engine/lib/balanced_ternary_engine/architectures.ex`, `pipeline.ex`, `cli.ex`, plus new converter modules and tests under `tools/balanced_ternary_engine/`.
- Affected docs: `details/13-model-conversion-guide.md`, `tools/balanced_ternary_engine/README.md`.
- No breaking changes to the `.tbin` format (version stays at 2).