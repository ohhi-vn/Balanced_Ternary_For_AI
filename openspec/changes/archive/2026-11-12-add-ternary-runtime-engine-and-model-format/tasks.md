# Tasks

## 1. Runtime Engine Hardening

- [x] 1.1 Implement robust `.tbin` loading into `Runtime` with structured errors and verify invalid files return `{:error, _}`
- [x] 1.2 Add configuration validation and verify invalid config is rejected without mutating state
- [x] 1.3 Implement cache reset and lifecycle helpers and verify prefill/decode/reset behavior
- [x] 1.4 Add deterministic execution tests and verify repeated runs match

## 2. Model Format Specification

- [x] 2.1 Document and validate `.tbin` header fields and verify version 1/2 files load
- [x] 2.2 Add format validation/round-trip tests and verify packed payloads reproduce
- [x] 2.3 Update exporter/CLI to report format version and verify `info` output

## 3. Documentation and Integration

- [x] 3.1 Update README and conversion guide with runtime/format usage and verify examples
- [x] 3.2 Run full test suite and escript build and verify no regressions

## Workflow follow-up

- Archive the change after review requirements are satisfied.