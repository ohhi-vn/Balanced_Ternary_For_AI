# Spec Delta

## ADDED Requirements

### Requirement: Format validation
The system SHALL validate `.tbin` structure and reject corrupt or incompatible files with structured errors.

#### Scenario: Corrupt file rejected
- **WHEN** a corrupt `.tbin` file is loaded
- **THEN** loading returns `{:error, _}`

### Requirement: Format compatibility
The system SHALL maintain backward compatibility with supported format versions.

#### Scenario: Older version loads
- **WHEN** a supported older `.tbin` version is loaded
- **THEN** it loads successfully or returns a clear compatibility error