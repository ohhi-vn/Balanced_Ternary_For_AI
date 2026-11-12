# Design

## Context

The Elixir app is under `tools/balanced_ternary_engine` and currently has minimal Mix metadata. Publishing to Hex.pm requires package metadata, docs, and a clean build. No runtime behavior changes are needed.

## Goals / Non-Goals

**Goals:**
- Make the app publishable as `ternary_converter` (or chosen name).
- Add HexDocs configuration and package files.
- Ensure tests and docs build.

**Non-Goals:**
- Changing public API.
- Publishing automatically (manual `mix hex.publish`).

## Decisions

- **Decision: Package the existing app in place.**
  - Rationale: avoids moving code and breaking paths.

- **Decision: Use ExDoc for documentation.**
  - Rationale: standard Elixir docs.

## Risks / Trade-offs

- [Package name availability] → verify with `mix hex.info` before publish.
- [Large deps] → Nx is required; acceptable.

## Migration Plan

Additive metadata only.

## Open Questions

None.