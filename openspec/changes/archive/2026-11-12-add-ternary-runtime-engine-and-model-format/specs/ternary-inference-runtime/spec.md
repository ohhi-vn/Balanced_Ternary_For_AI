# Spec Delta

## ADDED Requirements

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