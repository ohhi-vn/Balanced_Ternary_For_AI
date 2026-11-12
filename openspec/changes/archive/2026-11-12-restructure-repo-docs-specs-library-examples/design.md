# Design

## Context

The repo currently mixes `docs/details/`, `docs/examples/`, `tools/`, and top-level docs. Restructuring should preserve git history and update references.

## Goals / Non-Goals

**Goals:**
- Create a clear `docs/` hierarchy.
- Keep the Elixir library buildable.
- Update cross-references.

**Non-Goals:**
- Changing code behavior.

## Decisions

- **Decision: Move `docs/details/` to `docs/docs/details/` and `docs/examples/` to `docs/docs/examples/`.**
  - Rationale: conventional documentation layout.

- **Decision: Keep `tools/balanced_ternary_engine/` as the library root.**
  - Rationale: avoids breaking Mix paths.

## Risks / Trade-offs

- [Broken links] → update references.

## Migration Plan

Move directories with `git mv` and update links.

## Open Questions

None.