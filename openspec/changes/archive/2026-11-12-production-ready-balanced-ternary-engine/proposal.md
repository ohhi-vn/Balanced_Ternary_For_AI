# Proposal

## Why

The engine now covers conversion, runtime, format, and QAT, but production use requires stronger performance guarantees, robust error handling, utility APIs, and operational diagnostics.

## What Changes

- Add production-readiness utilities: validation, benchmarking helpers, and structured error reporting.
- Improve conversion performance and error handling for malformed inputs and large models.
- Improve runtime performance, deterministic execution, and cache safety.
- Extend model-format validation and compatibility checks.
- Extend QAT and runtime specs with production-grade requirements.

## Capabilities

### New Capabilities

- `balanced-ternary-production`: cross-cutting production-readiness requirements (performance, error handling, utilities, diagnostics).

### Modified Capabilities

- `ternary-conversion`: production-grade conversion performance, validation, and error handling.
- `ternary-inference-runtime`: runtime performance, cache safety, and deterministic execution.
- `ternary-model-format`: stricter validation and compatibility guarantees.
- `ternary-qat`: training robustness and freeze validation.

## Impact

- Affected code: `tools/balanced_ternary_engine/lib/**`, tests, docs.
- No breaking changes to `.tbin` format.