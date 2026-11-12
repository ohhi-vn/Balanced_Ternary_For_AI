# Proposal

## Why

The Elixir converter and runtime currently live under `tools/balanced_ternary_engine` with minimal Mix metadata, making them hard to publish, version, and consume as a standalone Hex.pm package. Packaging them properly enables reuse and distribution.

## What Changes

- Add complete Hex.pm package metadata (name, version, description, licenses, links, files, maintainers).
- Add documentation configuration and module docs suitable for HexDocs.
- Ensure the package builds cleanly (`mix compile`, `mix test`, `mix docs`, `mix hex.build`).
- Add a package-level README/changelog and publishing workflow guidance.
- Keep the library API stable; no runtime behavior changes.

## Capabilities

### New Capabilities

None (packaging/tooling change; no spec-level behavior changes).

### Modified Capabilities

None.

## Impact

- Affected files: `tools/balanced_ternary_engine/mix.exs`, `README.md`, new `CHANGELOG.md`, docs config.
- No changes to `.tbin` format or runtime behavior.