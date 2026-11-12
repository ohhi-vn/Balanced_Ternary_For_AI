# Spec Delta

## Purpose

Defines the contract for executing a converted `.tbin` model end-to-end, including layer chaining, attention, normalization, softmax, and KV-cached prefill/decode.

## ADDED Requirements

### Requirement: Layer chaining
The system SHALL execute a converted model by chaining its ternary and FP16 layers in stored order, applying each layer's forward pass to the previous output.

#### Scenario: Sequential forward pass
- **WHEN** a model with N layers is run on an input tensor
- **THEN** the output shape matches the final layer's output features and no layer is skipped

### Requirement: Attention block
The system SHALL compute multi-head attention using ternary Q/K/V/O projections and scaled dot-product attention.

#### Scenario: Attention produces correct shape
- **WHEN** an attention block runs on a sequence of tokens
- **THEN** the output has the same sequence length and hidden size as the input

### Requirement: Normalization and softmax
The system SHALL apply LayerNorm (or RMSNorm) and softmax where the architecture requires, operating in FP16/FP32.

#### Scenario: Softmax outputs sum to one
- **WHEN** softmax is applied to attention scores
- **THEN** each row of the result sums to 1 within floating-point tolerance

### Requirement: KV cache
The system SHALL maintain a key/value cache across decode steps and reuse cached keys/values for attention.

#### Scenario: Cached decode matches full recompute
- **WHEN** a sequence is decoded token-by-token with caching enabled
- **THEN** the output matches decoding the full sequence in one pass (within tolerance)

### Requirement: Prefill/decode split
The system SHALL distinguish a prefill phase (full sequence) from a decode phase (single new token) and use the cache appropriately in each.

#### Scenario: Prefill populates cache
- **WHEN** prefill runs on a prompt
- **THEN** the cache contains keys/values for every prompt token

#### Scenario: Decode uses cache
- **WHEN** decode runs for one new token
- **THEN** it attends to all prior cached keys/values plus the new token

### Requirement: Deterministic output
The system SHALL produce identical outputs for identical inputs and model state.

#### Scenario: Reproducibility
- **WHEN** the same model and input are run twice
- **THEN** the outputs are bitwise identical