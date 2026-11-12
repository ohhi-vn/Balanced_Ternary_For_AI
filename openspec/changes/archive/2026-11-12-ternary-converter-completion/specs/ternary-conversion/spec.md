# Spec Delta

## Purpose

Defines the contract for converting pretrained models into the `.tbin` balanced-ternary format, including PTQ quantization, mixed-precision policy, bias preservation, sparse packing, calibration, and validation reporting.

## ADDED Requirements

### Requirement: PTQ quantization modes
The system SHALL support three PTQ quantization modes — `delta` (fixed threshold with per-channel MSE-optimal scales), `twn` (threshold 0.7 × mean|W|), and `absmean` (BitNet b1.58 round-to-trit with shared scale) — and SHALL record the effective delta and per-channel scales in the output layer.

#### Scenario: TWN mode produces ternary weights
- **WHEN** a weight tensor is converted with mode `twn`
- **THEN** every output weight is in {-1, 0, +1} and the layer stores a non-empty per-channel scale list

#### Scenario: Absmean mode shares a scale across channels
- **WHEN** a weight tensor is converted with mode `absmean`
- **THEN** all per-channel scales equal the mean absolute weight of the tensor

### Requirement: Bias preservation
The system SHALL preserve every original bias term in the converted model, either by folding it into the per-channel scale or by storing it as an FP16 record, and SHALL NOT replace biases with zeros.

#### Scenario: Converted model retains biases
- **WHEN** a model containing bias tensors is converted
- **THEN** the resulting `.tbin` contains the bias values (folded or stored) and validation reports them

#### Scenario: No silent zeroing
- **WHEN** a layer has non-zero biases
- **THEN** the converted layer's bias list is not all zeros unless the source biases were all zero

### Requirement: Sharded safetensors input
The system SHALL read a model split across multiple `.safetensors` shards when given a shard index JSON, and SHALL surface a structured error (not raise) on malformed input or unsupported dtype.

#### Scenario: Multi-shard load succeeds
- **WHEN** conversion is given a valid shard index pointing to multiple safetensors files
- **THEN** all tensors from all shards are loaded and classified

#### Scenario: Malformed input returns error
- **WHEN** conversion receives a truncated or invalid safetensors file
- **THEN** the pipeline returns `{:error, _}` instead of raising

### Requirement: Convolution weights
The system SHALL convert 4-dimensional convolution weight tensors by reshaping them to 2D GEMM form, and SHALL NOT crash on such tensors.

#### Scenario: 4D conv weight converts
- **WHEN** a 4D conv weight tensor is encountered during conversion
- **THEN** it is converted to a ternary layer with a 2D shape and recorded in the output

### Requirement: Sparse packing selection
The system SHALL select sparse packing when layer sparsity exceeds the configured threshold (default ~88%) and dense packing otherwise, and SHALL report which packing was used.

#### Scenario: High-sparsity layer uses sparse packing
- **WHEN** a layer's sparsity exceeds the threshold
- **THEN** its stored payload uses sparse encoding and round-trips exactly

#### Scenario: Low-sparsity layer uses dense packing
- **WHEN** a layer's sparsity is below the threshold
- **THEN** its stored payload uses dense 10-trits-per-word encoding

### Requirement: Calibration options
The system SHALL accept `--sparsity-target`, `--clip`, and `--calibration-data` options on real-model conversion and apply them per layer.

#### Scenario: Sparsity target honored
- **WHEN** conversion is run with a sparsity target
- **THEN** each ternary layer's sparsity is within a documented tolerance of the target (or the achievable bound is reported)

#### Scenario: Outlier clipping applied
- **WHEN** conversion is run with `--clip`
- **THEN** scale factors are computed from clipped weights

### Requirement: Mixed-precision policy
The system SHALL classify each tensor as ternary, keep-FP16, or skip based on architecture policy, and SHALL additionally promote the most sensitive layers to FP16 when an accuracy target is supplied.

#### Scenario: Sensitivity promotion
- **WHEN** an accuracy target is supplied
- **THEN** the least-accurate-under-ternary layers are promoted to FP16 until the target is met (or all are promoted)

### Requirement: Validation reporting
The system SHALL report round-trip correctness, per-layer sparsity, output cosine similarity (when a reference is available), and model size/compression for every converted model.

#### Scenario: CLI validate runs full harness
- **WHEN** `validate` is invoked on a `.tbin` file
- **THEN** it reports round-trip, sparsity, similarity (if reference provided), and size statistics

### Requirement: Mix tasks and escript
The system SHALL provide `mix convert`, `mix infer`, and `mix analyze` tasks and a buildable escript, and SHALL document the correct tool path (`tools/balanced_ternary_engine`).

#### Scenario: mix convert works
- **WHEN** a user runs `mix convert --input model.safetensors --output model.tbin`
- **THEN** a valid `.tbin` is produced and a summary is printed