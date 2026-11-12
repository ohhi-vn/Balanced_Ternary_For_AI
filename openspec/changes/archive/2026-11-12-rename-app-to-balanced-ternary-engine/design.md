# Design

## Context

The codebase uses `TernaryConverter` as the root module and `:ternary_converter` as the Mix app. A full rename touches all modules, tests, and docs.

## Goals / Non-Goals

**Goals:**
- Rename modules and app consistently.
- Keep tests passing.

**Non-Goals:**
- Changing behavior or APIs beyond naming.

## Decisions

- **Decision: Use sed-based mechanical rename across lib/test/guides.**
  - Rationale: consistent and low-risk.

- **Decision: Keep file paths under `tools/balanced_ternary_engine` but update module names.**
  - Rationale: avoids moving files.

## Risks / Trade-offs

- [Missed references] → run compile and tests.

## Migration Plan

Mechanical rename, then verify.

## Open Questions

None.