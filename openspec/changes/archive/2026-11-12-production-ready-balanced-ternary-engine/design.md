# Design

## Context

The engine covers conversion, runtime, format, and QAT. Production readiness requires better performance, diagnostics, and error handling across these areas.

## Goals / Non-Goals

**Goals:**
- Add utility and diagnostic APIs.
- Improve error handling and performance characteristics.
- Keep changes backward compatible.

**Non-Goals:**
- Rewriting the core algorithms.
- Changing the `.tbin` format version.

## Decisions

- **Decision: Add utilities as new modules/functions rather than changing core APIs.**
  - Rationale: minimizes breakage.

- **Decision: Prefer structured errors over exceptions.**
  - Rationale: consistent with existing patterns.

## Risks / Trade-offs

- [Performance gains may be limited] → benchmark and iterate.
- [Added API surface] → document thoroughly.

## Migration Plan

Additive only.

## Open Questions

None.