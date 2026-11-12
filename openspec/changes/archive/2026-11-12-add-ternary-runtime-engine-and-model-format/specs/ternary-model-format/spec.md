# Spec Delta

## Purpose

Defines the `.tbin` balanced-ternary model format, including its header, mixed ternary/FP16 records, versioning, and validation rules for interoperable storage and loading.

## ADDED Requirements

### Requirement: Format header
The `.tbin` format SHALL begin with a magic number, format version, layer count, and metadata length followed by UTF-8 JSON metadata.

#### Scenario: Header parsed
- **WHEN** a valid `.tbin` file is loaded
- **THEN** its magic, version, layer count, and metadata are read successfully

#### Scenario: Bad magic rejected
- **WHEN** a file with an invalid magic number is loaded
- **THEN** loading returns a structured error

### Requirement: Mixed record kinds
The format SHALL support ternary records and raw FP16 records within the same file, each identified by a record kind discriminator.

#### Scenario: Mixed file loads
- **WHEN** a `.tbin` file contains both ternary and FP16 records
- **THEN** all records are loaded with their correct kind

### Requirement: Version compatibility
The loader SHALL read version 1 ternary-only files and version 2 mixed files, and SHALL reject unsupported versions with a structured error.

#### Scenario: Version 1 readable
- **WHEN** a version 1 `.tbin` file is loaded
- **THEN** it is interpreted as ternary-only records

#### Scenario: Unsupported version rejected
- **WHEN** a `.tbin` file declares an unsupported version
- **THEN** loading returns a structured error

### Requirement: Round-trip validation
The format SHALL preserve weight data such that packing and unpacking a ternary payload reproduces the stored bytes exactly.

#### Scenario: Ternary payload round-trips
- **WHEN** a ternary record is unpacked and repacked
- **THEN** the resulting bytes equal the stored payload

### Requirement: Metadata presence
Every `.tbin` file SHALL contain parseable JSON metadata, even if empty.

#### Scenario: Empty metadata allowed
- **WHEN** a `.tbin` file is written with an empty metadata map
- **THEN** loading succeeds and returns an empty metadata map