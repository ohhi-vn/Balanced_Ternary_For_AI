# Proposal

## Why

The repository has a minimal `TernaryConverter.Runtime` module and a documented `.tbin` format, but neither is backed by a complete, testable contract. The runtime lacks robust loading, configuration validation, cache lifecycle handling, and error reporting, while the binary format is only described in prose. Formalizing both makes converted models executable and interoperable.

## What Changes

- Implement a robust balanced-ternary runtime engine that loads `.tbin` models with configuration, runs deterministic forward passes, applies attention/norm/softmax, and manages KV-cache prefill/decode.
- Add a dedicated model-format specification for `.tbin` (header, mixed record kinds, fields, versioning, validation, and compatibility).
- Extend the inference-runtime spec with runtime-engine requirements (loading, configuration, error handling, cache lifecycle).
- Add tests covering runtime loading, deterministic execution, cache behavior, and format round-trips.

## Capabilities

### New Capabilities

- `ternary-model-format`: specification of the `.tbin` model format, including header, layer records, versioning, and validation.

### Modified Capabilities

- `ternary-inference-runtime`: extend runtime requirements for model loading, configuration validation, cache lifecycle, and deterministic execution.

## Impact

- Affected code: `tools/balanced_ternary_engine/lib/balanced_ternary_engine/runtime.ex`, `exporter.ex`, `cli.ex`, and tests.
- Affected docs: `details/13-model-conversion-guide.md` and `tools/balanced_ternary_engine/README.md`.
- No breaking changes to `.tbin` version 2; format spec documents existing behavior.