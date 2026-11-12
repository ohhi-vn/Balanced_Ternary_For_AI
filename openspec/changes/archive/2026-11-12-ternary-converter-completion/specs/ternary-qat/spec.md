# Spec Delta

## Purpose

Defines the contract for the quantization-aware training path that recovers accuracy lost by PTQ, producing a frozen `.tbin` model via straight-through estimation, distillation, and sparsity regularization.

## ADDED Requirements

### Requirement: QAT layer with shadow weights
The system SHALL maintain FP32 shadow weights that are ternarized on each forward pass, and SHALL expose the effective ternary weights for inference while retaining shadow weights for gradient updates.

#### Scenario: Forward uses ternary weights
- **WHEN** a QAT layer runs a forward pass
- **THEN** the computation uses ternarized weights (not the raw shadow weights)

#### Scenario: Shadow weights persist
- **WHEN** a QAT layer is updated
- **THEN** the shadow weights change and the ternary weights are recomputed on the next forward pass

### Requirement: Straight-through estimator
The system SHALL allow gradients to flow to shadow weights through the non-differentiable ternarization step.

#### Scenario: Gradients reach shadow weights
- **WHEN** a training step computes a loss and backpropagates
- **THEN** the shadow weights receive a non-zero gradient update

### Requirement: Knowledge distillation
The system SHALL support an optional FP32 teacher model whose outputs are used as a soft target in the training loss.

#### Scenario: Distillation loss applied
- **WHEN** a teacher is provided
- **THEN** the training loss includes a term matching the teacher's outputs

### Requirement: Sparsity regularization
The system SHALL support a configurable sparsity regularization term that encourages zero weights.

#### Scenario: Regularization affects sparsity
- **WHEN** sparsity regularization is enabled with a positive coefficient
- **THEN** the resulting frozen model's sparsity is no lower than without regularization (within tolerance)

### Requirement: Gradual delta scheduling
The system SHALL support a schedule that increases the ternarization threshold over training according to a configured ramp.

#### Scenario: Delta increases over epochs
- **WHEN** a ramp schedule is configured
- **THEN** the effective delta at the final epoch exceeds the initial delta

### Requirement: Freeze to .tbin
The system SHALL convert a trained QAT layer into a standard ternary `.tbin` layer with recorded delta and scales.

#### Scenario: Frozen layer is loadable
- **WHEN** a QAT layer is frozen and exported
- **THEN** the resulting `.tbin` loads via the standard loader and passes round-trip validation