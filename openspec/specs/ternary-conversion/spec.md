# ternary-conversion Specification

## Purpose

Defines the contract for converting pretrained models into the `.tbin` balanced-ternary format, including PTQ quantization, mixed-precision policy, bias preservation, sparse packing, calibration, and validation reporting.

## Requirements

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

### Requirement: YOLO model conversion
The system SHALL provide a YOLO converter tool that converts YOLO-family checkpoints (backbone, neck, and detection head) into `.tbin`, reshaping 4D convolution weights to 2D GEMM form, ternarizing backbone convolution/linear weights, and storing detection heads and normalization tensors as FP16 records.

#### Scenario: YOLO conv weights convert
- **WHEN** a YOLO checkpoint containing 4D convolution weights is converted
- **THEN** those weights become ternary layers with 2D shapes and no error is raised

#### Scenario: Detection head preserved as FP16
- **WHEN** a YOLO checkpoint's detection-head tensors are classified
- **THEN** they are stored as FP16 records rather than ternary layers

#### Scenario: YOLO conversion produces valid .tbin
- **WHEN** the YOLO converter tool runs on a valid checkpoint
- **THEN** it writes a loadable `.tbin` and prints a summary with layer counts and sparsity

### Requirement: ImageNet model conversion
The system SHALL provide an ImageNet converter tool that converts image-classification backbones (ResNet/MobileNet/EfficientNet/ViT-style) into `.tbin`, ternarizing convolution/linear backbone weights and storing the classifier head and normalization tensors as FP16 records.

#### Scenario: ImageNet backbone converts
- **WHEN** an ImageNet-style checkpoint is converted
- **THEN** backbone weights become ternary layers and classifier-head tensors remain FP16

#### Scenario: ImageNet conversion reports stats
- **WHEN** the ImageNet converter tool completes
- **THEN** it reports total parameters, sparsity, and compression ratio

### Requirement: LLM under 8B conversion
The system SHALL provide an LLM converter tool targeting decoder-only models under 8B parameters that reads sharded safetensors checkpoints, applies the decoder mixed-precision policy, and produces a `.tbin` with perplexity/cosine-similarity validation hooks.

#### Scenario: Sharded LLM checkpoint converts
- **WHEN** the LLM converter tool is given a shard index for a sub-8B model
- **THEN** all shards are loaded and converted without raising

#### Scenario: Decoder sensitive tensors preserved
- **WHEN** embeddings, norms, and LM head tensors are encountered
- **THEN** they are stored as FP16 records rather than ternary layers

#### Scenario: LLM conversion validates output
- **WHEN** the LLM converter tool completes
- **THEN** it reports round-trip correctness and output cosine similarity when a reference is available

### Requirement: Converter tool CLI entry points
The system SHALL expose the YOLO, ImageNet, and LLM converters through the CLI as selectable profiles or commands with documented options, and SHALL reject unknown profiles with a clear error.

#### Scenario: Profile selection works
- **WHEN** a user invokes the CLI with a supported profile name
- **THEN** the corresponding converter runs and produces a `.tbin`

#### Scenario: Unknown profile rejected
- **WHEN** a user invokes the CLI with an unsupported profile name
- **THEN** the tool prints a usage error and exits non-zero without writing output

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
