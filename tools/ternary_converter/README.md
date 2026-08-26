# TernaryConverter

A complete Elixir application for converting neural network weights to balanced ternary format `{-1, 0, +1}`.

## Features

- **Safetensors input**: Read HuggingFace `.safetensors` checkpoints directly in pure Elixir (f32/f16/bf16/i64) — no Python
- **Architecture policies**: `llama` / `gpt2` / `bert` / `vit` / `generic` mixed-precision rules; GPT-2 Conv1D transposition handled
- **Quantization modes**: `twn` (Δ = 0.7·mean|W|), `absmean` (BitNet b1.58), `delta` (fixed threshold + MSE-optimal scales)
- **Per-channel scaling**: least-squares optimal scale factors for accuracy recovery
- **Binary packing**: 10 trits per 16-bit word (base-3 encoding)
- **Sparse encoding**: Index+sign format for high-sparsity layers
- **Sensitivity analysis**: Per-layer quantization sensitivity measurement
- **Mixed precision**: Sensitive tensors (embeddings, norms, LM head) stored as FP16 records in `.tbin` v2
- **CLI tool**: Convert, demo, info, and validate commands

## Installation

```bash
cd tools/ternary_converter
mix deps.get
```

## Quick Start

### Convert a real checkpoint (safetensors → .tbin)

```bash
mix run -e 'TernaryConverter.CLI.main(["convert", "--input", "model.safetensors", "--arch", "llama", "--mode", "absmean"])'
```

Options:
- `--arch llama|gpt2|bert|vit|generic` — mixed-precision policy
- `--mode twn|absmean|delta` — quantization rule (`--delta F` sets the threshold for `delta`)
- `--output PATH` — output file (default: `<input basename>.tbin`)

Then inspect and validate:

```bash
mix run -e 'TernaryConverter.CLI.main(["info", "--model", "model.tbin"])'
mix run -e 'TernaryConverter.CLI.main(["validate", "--model", "model.tbin"])'
```

### Run the demo

```bash
mix run -e "TernaryConverter.CLI.main([\"demo\"])"
```

### Convert synthetic model to .tbin

```bash
mix run -e "TernaryConverter.CLI.main([\"convert\", \"--delta\", \"0.5\", \"--output\", \"model.tbin\"])"
```

### Inspect a .tbin file

```bash
mix run -e "TernaryConverter.CLI.main([\"info\", \"--model\", \"model.tbin\"])"
```

### Validate a .tbin file

```bash
mix run -e "TernaryConverter.CLI.main([\"validate\", \"--model\", \"model.tbin\"])"
```

### Run as escript

```bash
mix escript.build
./ternary_converter demo --rows 128 --cols 256 --delta 0.5
./ternary_converter convert --delta 0.5 --output my_model.tbin
./ternary_converter info --model my_model.tbin
```

## Library Usage

```elixir
# End-to-end: real safetensors checkpoint → .tbin
{:ok, summary} =
  TernaryConverter.Pipeline.run("model.safetensors", "model.tbin",
    arch: :llama,
    mode: :absmean
  )

summary.ternary_layers   # [%Layer{}] — packed trits + scales
summary.fp16_layers      # [%RawLayer{}] — kept dense (embeddings, norms, lm_head)
summary.stats            # %{sparsity: ..., compression_ratio: ...}

# Create a ternary layer from weights
w = Nx.random_uniform({256, 512})
layer = TernaryConverter.convert_layer(w, "fc1", delta: 0.5)

# Run inference
input = Nx.random_uniform({1, 512})
output = TernaryConverter.inference([layer], input)

# Convert a full model
weights = %{
  "fc1" => Nx.random_uniform({256, 512}),
  "fc2" => Nx.random_uniform({128, 256}),
  "fc3" => Nx.random_uniform({64, 128})
}
layers = TernaryConverter.convert_all(weights, delta: 0.5)

# Analyze sensitivity
sample_input = Nx.random_uniform({1, 512})
sensitivities = TernaryConverter.analyze_sensitivity(weights, sample_input)

# Auto mixed precision
{assignment, n_ternary, n_fp16} =
  TernaryConverter.auto_mixed_precision(weights, sample_input, 0.5, 0.99)

# Export to .tbin
:ok = TernaryConverter.export(layers, "model.tbin",
  metadata: %{model_name: "my_model", version: "1.0.0"}
)

# Load from .tbin
{:ok, loaded_layers, metadata} = TernaryConverter.load("model.tbin")

# Get statistics
stats = TernaryConverter.stats(layers)
# %{num_layers: 3, total_parameters: ..., compression_ratio: ...}
```

## Running Tests

```bash
mix test
```

## Project Structure

```
ternary_converter/
├── lib/
│   ├── ternary_converter.ex              # Main API
│   └── ternary_converter/
│       ├── quantizer.ex                  # Modes delta|twn|absmean, auto-threshold
│       ├── packer.ex                     # Binary packing (10→16, sparse)
│       ├── layer.ex                      # Ternary layer struct + forward pass
│       ├── raw_layer.ex                  # FP16 records for sensitive tensors
│       ├── exporter.ex                   # .tbin v2 export/import (mixed)
│       ├── safetensors.ex                # Pure-Elixir safetensors reader/writer
│       ├── architectures.ex              # llama/gpt2/bert/vit policies
│       ├── pipeline.ex                   # End-to-end conversion
│       ├── sensitivity.ex                # Per-layer sensitivity analysis
│       ├── stats.ex                      # Aggregate statistics
│       └── cli.ex                        # Command-line interface
├── test/
│   ├── quantizer_test.exs
│   ├── packer_test.exs
│   ├── layer_test.exs
│   ├── safetensors_test.exs
│   ├── architectures_test.exs
│   ├── pipeline_test.exs
│   └── integration_test.exs
├── mix.exs
└── README.md
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `TernaryConverter` | Main API: convert, inference, export, stats, validate |
| `TernaryConverter.Pipeline` | End-to-end: safetensors → classify → quantize → .tbin |
| `TernaryConverter.Quantizer` | `quantize/2` modes, `twn_delta/1`, `auto_threshold/2`, `quality_metrics/2` |
| `TernaryConverter.Safetensors` | Pure-Elixir `.safetensors` reader/writer |
| `TernaryConverter.Architectures` | Per-architecture mixed-precision classification + transforms |
| `TernaryConverter.Packer` | `pack/1`, `unpack/2`, `pack_sparse/1`, `compression_ratio/1` |
| `TernaryConverter.Layer` | `from_dense/3`, `forward/2`, `forward_nx/2`, `to_binary/1` |
| `TernaryConverter.RawLayer` | FP16 dense record for embeddings/norms/LM head |
| `TernaryConverter.Exporter` | `export/3`, `load/1` for .tbin v1/v2 format |
| `TernaryConverter.Sensitivity` | `analyze/3`, `auto_mixed_precision/4` |
| `TernaryConverter.Stats` | Aggregate size/sparsity statistics over mixed layers |
| `TernaryConverter.CLI` | CLI: convert, demo, info, validate |

## .tbin Format

Binary format for storing ternary models (version 2):

```
[Header][Record 1][Record 2]...[Record N]

Header:
  magic:       "TBN\0" (4 bytes)
  version:     uint32 (2)
  num_layers:  uint32
  meta_len:    uint32
  metadata:    JSON (meta_len bytes)

Per record:
  kind:        uint8  (0 = ternary, 1 = raw FP16)
  size:        uint32
  payload:
    ternary:
      name_len:    uint16
      name:        UTF-8 bytes
      out_feat:    uint32
      in_feat:     uint32
      delta:       float32
      sparsity:    float32
      scales:      num_scales × float16
      biases:      num_biases × float16
      weights:     packed trits (10 per 16-bit word)
    raw fp16:
      name_len:    uint16
      name:        UTF-8 bytes
      ndims:       uint8
      dims:        ndims × uint32
      data:        flat f16 payload

Version 1 files (ternary-only, no kind byte) remain readable.
```

## License

See the project LICENSE file.
