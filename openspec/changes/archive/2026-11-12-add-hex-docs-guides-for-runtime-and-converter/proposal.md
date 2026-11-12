# Proposal

## Why

HexDocs currently exposes only API reference. Users need task-oriented guides for converting models and running them at runtime, published alongside the package.

## What Changes

- Add ExDoc guides for conversion workflows and runtime usage.
- Wire guides into `mix.exs` docs configuration.
- Ensure guides build and are included in the Hex package.

## Capabilities

### New Capabilities

None (documentation only).

### Modified Capabilities

None.

## Impact

- Affected files: `tools/balanced_ternary_engine/mix.exs`, new guide files under `guides/`.
- No code or format changes.