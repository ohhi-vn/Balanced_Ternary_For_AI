# Design

## Context

ExDoc is already configured. Guides should be Markdown files referenced in `docs/0` extras/groups.

## Goals / Non-Goals

**Goals:**
- Add conversion and runtime guides.
- Include them in HexDocs.

**Non-Goals:**
- API reference changes.

## Decisions

- **Decision: Place guides under `guides/` and reference in `mix.exs`.**
  - Rationale: standard ExDoc layout.

## Risks / Trade-offs

- [Guide drift] → keep examples minimal and tested.

## Migration Plan

Additive only.

## Open Questions

None.