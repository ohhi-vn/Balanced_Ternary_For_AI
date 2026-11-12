# Spec Delta

## ADDED Requirements

### Requirement: Conversion error handling
The system SHALL return structured errors for malformed safetensors, unsupported dtypes, and invalid options.

#### Scenario: Malformed input returns error
- **WHEN** conversion receives invalid input
- **THEN** it returns `{:error, _}` instead of raising

### Requirement: Conversion performance
The system SHALL process large models without excessive memory growth and report progress.

#### Scenario: Large model converts
- **WHEN** conversion runs on a large checkpoint
- **THEN** it completes and reports layer counts