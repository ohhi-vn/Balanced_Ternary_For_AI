# Spec Delta

## ADDED Requirements

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