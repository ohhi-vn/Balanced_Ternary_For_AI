# Design

## Context

The runtime module already provides layer chaining, softmax, layer norm, attention, and prefill/decode, but loading and validation are thin and the `.tbin` format is only documented in prose. See `proposal.md`.

## Goals / Non-Goals

**Goals:**
- Harden runtime loading and configuration handling.
- Formalize `.tbin` format behavior and validation.
- Keep implementation additive and backward compatible with v1/v2 files.

**Non-Goals:**
- Distributed serving or batching.
- New model architectures beyond generic layer chaining.

## Decisions

- **Decision: Keep `.tbin` version 2 as canonical; retain v1 read support.**
  - Rationale: avoids breaking existing files.

- **Decision: Runtime remains a struct wrapping layers, config, and cache.**
  - Rationale: minimal change to existing API.

- **Decision: Validation returns structured errors rather than raising.**
  - Rationale: matches converter conventions.

## Risks / Trade-offs

- [Attention implementation is simplified] → document as minimal runtime; extend later.
- [Format spec may diverge from docs] → update docs alongside implementation.

## Migration Plan

Additive. Existing `.tbin` files remain readable.

## Open Questions

None.