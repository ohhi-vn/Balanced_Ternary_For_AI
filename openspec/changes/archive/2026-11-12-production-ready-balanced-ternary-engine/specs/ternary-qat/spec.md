# Spec Delta

## ADDED Requirements

### Requirement: QAT robustness
The system SHALL handle invalid training inputs and configurations with structured errors.

#### Scenario: Invalid QAT input rejected
- **WHEN** QAT receives invalid input
- **THEN** it returns `{:error, _}` instead of raising

### Requirement: Freeze validation
The system SHALL validate frozen QAT output before export.

#### Scenario: Frozen model validates
- **WHEN** a QAT layer is frozen and exported
- **THEN** the resulting `.tbin` loads and passes round-trip validation