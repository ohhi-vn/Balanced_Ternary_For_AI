# ternary-inference-runtime Specification

## Purpose

Defines the contract for executing a converted `.tbin` model end-to-end, including layer chaining, attention, normalization, softmax, and KV-cached prefill/decode.

## Requirements

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

### Requirement: Runtime model loading
The system SHALL load a `.tbin` model into a runtime instance with an optional configuration map, and SHALL return a structured error when the file is missing, corrupt, or incompatible.

#### Scenario: Valid model loads
- **WHEN** a valid `.tbin` file is loaded into the runtime
- **THEN** the runtime contains the model layers and an empty cache

#### Scenario: Invalid model returns error
- **WHEN** the runtime is asked to load a missing or corrupt `.tbin` file
- **THEN** it returns `{:error, _}` instead of raising

### Requirement: Runtime configuration validation
The system SHALL validate runtime configuration values and reject invalid configuration without mutating model state.

#### Scenario: Invalid config rejected
- **WHEN** a runtime is created with an invalid configuration value
- **THEN** loading fails with a structured error and no partial runtime is returned

### Requirement: Cache lifecycle
The system SHALL populate the cache during prefill, reuse cached keys and values during decode, and provide a way to reset the cache without reloading the model.

#### Scenario: Prefill populates cache
- **WHEN** prefill runs on a prompt
- **THEN** the cache contains keys and values for every prompt token

#### Scenario: Reset clears cache
- **WHEN** the cache is reset
- **THEN** subsequent decode starts from an empty cache

### Requirement: Deterministic runtime execution
The system SHALL produce identical outputs for identical model state, input, and cache contents.

#### Scenario: Repeated run matches
- **WHEN** the same runtime and input are executed twice
- **THEN** the outputs are equal

### Requirement: Runtime performance
The system SHALL execute forward passes efficiently and avoid unnecessary allocations.

#### Scenario: Repeated forward runs
- **WHEN** the runtime executes repeated forward passes
- **THEN** outputs remain correct and stable

### Requirement: Runtime error handling
The system SHALL return structured errors for invalid model files and configurations.

#### Scenario: Invalid runtime config rejected
- **WHEN** an invalid configuration is supplied
- **THEN** runtime loading returns `{:error, _}`
