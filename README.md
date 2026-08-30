# Balanced Ternary for Neural Network Inference

A comprehensive exploration of balanced ternary (`{-1, 0, +1}`) as a weight representation for neural network accelerators — spanning quantization theory, training methodology, memory packing, hardware architecture, LLM inference engine design, model conversion tools, and a 48-week research roadmap.

## Why Balanced Ternary

| Property | FP32 | INT8 | Ternary |
|----------|------|------|---------|
| Bits per weight | 32 | 8 | ~1.585 |
| Model size (1B params) | 4 GB | 1 GB | ~200 MB |
| Multiply cost | Full FP32 MUL | INT8 MUL | Add/sub/skip |
| Sparsity support | No | No | Natural (zero weight) |

The zero trit is the key advantage: it encodes sparsity directly, turning near-zero weights into skipped computation.

## Contents

### Concept & Architecture

| File | Topic |
|------|-------|
| [`concept.md`](concept.md) | High-level overview, storage efficiency, packing, hybrid architecture, LLM accelerator vision |
| [`details/01-architecture-overview.md`](details/01-architecture-overview.md) | Hybrid precision model, layer mapping, data flow, scaling strategies, software stack |
| [`details/05-hardware-architecture.md`](details/05-hardware-architecture.md) | Systolic PE array, decoder, memory hierarchy, power estimates, ISA, verification |
| [`details/06-llm-inference-engine.md`](details/06-llm-inference-engine.md) | Prefill/decode, KV cache, throughput estimation, batch processing, energy analysis |

### Quantization & Training

| File | Topic |
|------|-------|
| [`details/02-weight-quantization.md`](details/02-weight-quantization.md) | Quantizer design, threshold selection, per-channel scaling, QAT, error analysis |
| [`details/03-training.md`](details/03-training.md) | Training pipeline, STE variants, sparsity regularization, distillation, distributed training |
| [`details/07-accuracy-analysis.md`](details/07-accuracy-analysis.md) | Benchmarks, sensitivity analysis, ablation studies, reproducibility checklist |

### Storage & Encoding

| File | Topic |
|------|-------|
| [`details/04-storage-format.md`](details/04-storage-format.md) | 5→8 / 10→16 packing, sparse encoding, hybrid storage, bandwidth calculation |
| [`details/09-differential-encoding.md`](details/09-differential-encoding.md) | Two-wire differential trit encoding, wire-swap negation, CMOS compatibility, DFT |

### Hardware & Deployment

| File | Topic |
|------|-------|
| [`details/11-current-hardware-gpu-cpu-npu.md`](details/11-current-hardware-gpu-cpu-npu.md) | Running ternary models on GPU, CPU, NPU, FPGA today — CUDA kernels, AMX, HVX, Vulkan |
| [`details/12-custom-ternary-accelerator-design.md`](details/12-custom-ternary-accelerator-design.md) | Full ASIC architecture spec: PE design, memory subsystem, ISA, compiler, cost analysis |

### Conversion & Tools

| File | Topic |
|------|-------|
| [`details/13-model-conversion-guide.md`](details/13-model-conversion-guide.md) | PTQ/QAT pipelines, model-specific recipes, validation suite, Elixir conversion toolchain |
| [`tools/ternary_converter/`](tools/ternary_converter/) | Full Elixir application: convert (safetensors → .tbin), demo, info, validate CLI + library API |
| [`examples/ternary_accel.livemd`](examples/ternary_accel.livemd) | Ternary GEMM on NVIDIA CUDA (EXLA) and Apple Silicon MLX (EMLX) via Nx — Livebook |

### Feasibility Analysis

| File | Topic |
|------|-------|
| [`details/14-ternary-llm-feasibility.md`](details/14-ternary-llm-feasibility.md) | Ternary for trillion-parameter LLMs: scaling laws, MoE, cost analysis |
| [`details/15-ternary-vision-computing.md`](details/15-ternary-vision-computing.md) | Ternary for vision: CNNs, ViT, detection, segmentation, edge deployment |
| [`details/16-fpga-experiment-guide.md`](details/16-fpga-experiment-guide.md) | FPGA experiment guide: board selection, RTL design, benchmarking |
| [`details/17-asic-implementation-guide.md`](details/17-asic-implementation-guide.md) | ASIC implementation guide: RTL design, physical design, DFT, manufacturing |

### Comparison & Roadmap

| File | Topic |
|------|-------|
| [`details/10-comparison.md`](details/10-comparison.md) | Detailed comparison with BNN, INT8, INT4, FP8, pruning — pros/cons, decision matrix, TCO |
| [`details/08-research-roadmap.md`](details/08-research-roadmap.md) | 48-week roadmap from software simulation to ASIC tape-out, publication strategy |

### Theory & Analysis Deep Dives

| File | Topic |
|------|-------|
| [`details/18-multiplication-as-selection.md`](details/18-multiplication-as-selection.md) | Why ternary removes the multiplier: MAC cost model, select/add/skip datapath, op-count math |
| [`details/19-sparsity-economics.md`](details/19-sparsity-economics.md) | Where zeros come from, accuracy/compute tradeoffs, sparse formats, hardware gating |
| [`details/20-quantization-theory.md`](details/20-quantization-theory.md) | Derivations: least-squares scales, TWN joint optimum, absmean (BitNet b1.58), SQNR, bisection |
| [`details/21-memory-bandwidth-analysis.md`](details/21-memory-bandwidth-analysis.md) | Roofline at batch 1, SRAM fit arithmetic, energy breakdown, batch effects |
| [`details/22-history-and-vision.md`](details/22-history-and-vision.md) | Setun history, why binary won, target markets, phase gates, honest risks |
| [`details/23-practical-labs-and-exam.md`](details/23-practical-labs-and-exam.md) | Guided study path: 5 hands-on labs + capstone exam (answer key, grading rubric) |

### Examples

| Notebook | Content |
|----------|---------|
| [`examples/ternary_pure.livemd`](examples/ternary_pure.livemd) | Pure Elixir Livebook: quantize → pack → forward pass → differential encoding |
| [`examples/ternary_nx.livemd`](examples/ternary_nx.livemd) | Nx-based Livebook: tensor quantization → GEMM → sparsity analysis → packing → QAT simulation |
| [`examples/ternary_accel.livemd`](examples/ternary_accel.livemd) | GPU acceleration Livebook: ternary GEMM on CUDA (EXLA) and Apple Silicon (MLX), packed-trit decoding on device, throughput benchmark |

## Target Application

A **balanced-ternary transformer inference accelerator** for edge devices — 2–10 W, 100M–1B parameter models fitting entirely in on-chip SRAM, with throughput of ~50K tokens/s decode.

## Quick Start

### Convert a real model to ternary

```bash
cd tools/ternary_converter
mix deps.get

# HuggingFace safetensors → .tbin, LLaMA policy, BitNet-style absmean mode
mix run -e 'TernaryConverter.CLI.main(["convert", "--input", "model.safetensors", "--arch", "llama", "--mode", "absmean"])'

# Inspect and validate
mix run -e 'TernaryConverter.CLI.main(["info", "--model", "model.tbin"])'
mix run -e 'TernaryConverter.CLI.main(["validate", "--model", "model.tbin"])'
```

Architectures: `llama` · `gpt2` · `bert` · `vit` · `generic` (mixed-precision policies). Modes: `twn` (default), `absmean`, `delta`.

### Run a demo

```bash
mix run -e 'TernaryConverter.CLI.main(["demo"])'

# Or convert a synthetic model without an input file
mix run -e 'TernaryConverter.CLI.main(["convert", "--sparsity", "0.5", "--output", "model.tbin"])'
```

### Run on CUDA / Apple MLX

Open [`examples/ternary_accel.livemd`](examples/ternary_accel.livemd) in [Livebook](https://livebook.dev). The setup cell installs Nx; add `{:exla, "~> 0.12"}` (CUDA) or `{:emlx, "~> 0.4"}` (Apple Silicon) for GPU backends.

Decodes packed trits on-device and benchmarks ternary vs FP32 GEMM. The notebook covers backend selection.

### Run the pure Elixir example

Open [`examples/ternary_pure.livemd`](examples/ternary_pure.livemd) in [Livebook](https://livebook.dev) — no dependencies required.

### Run the Nx-based example

Open [`examples/ternary_nx.livemd`](examples/ternary_nx.livemd) in [Livebook](https://livebook.dev) — the setup cell installs Nx.

## License

```
Copyright 2026 Balanced Ternary Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
