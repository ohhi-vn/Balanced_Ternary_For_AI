# Proposal

## Why

The repository mixes research notes, examples, and the Elixir library at the top level, making navigation and publishing harder. A clearer layout improves discoverability and maintainability.

## What Changes

- Move long-form research notes into a `docs/` directory.
- Move examples into `docs/docs/examples/` or `docs/examples/` with clearer naming.
- Keep the Elixir library under `tools/balanced_ternary_engine/` (or promote to root if desired).
- Update cross-references in README and guides.

## Capabilities

### New Capabilities

None (repository organization only).

### Modified Capabilities

None.

## Impact

- Affected paths: `docs/details/`, `docs/examples/`, `tools/`, `README.md`, docs.
- No code or spec behavior changes.