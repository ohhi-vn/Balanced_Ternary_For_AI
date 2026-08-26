# TernaryAccel — Balanced Ternary on Nx (CUDA / Apple MLX)

Runs the balanced-ternary pipeline **on GPU** through Nx:

- quantizes weights on-device with the BitNet b1.58 **absmean** rule
- stores trits as packed `u8` codes `{0,1,2}` → 4× less weight memory than FP32
- decodes trits **on device** inside a compiled kernel and runs GEMM — the software analogue of the hardware add/sub/skip decoder from [`details/05-hardware-architecture.md`](../../details/05-hardware-architecture.md)
- verifies accuracy against the FP32 reference and benchmarks dense vs ternary GEMM

## Backends

| Backend | Library | Requirements | Select |
|---------|---------|--------------|--------|
| NVIDIA CUDA | [EXLA](https://hex.pm/packages/exla) | NVIDIA GPU + CUDA build of XLA | `backend: :cuda` |
| Apple Silicon | [EMLX](https://hex.pm/packages/emlx) (Apple MLX) | macOS on M-series | `backend: :mlx` |
| XLA CPU | EXLA | any | `backend: :exla_cpu` |
| CPU fallback | Nx.BinaryBackend | any (always available) | `backend: :binary` |

`TernaryAccel.run()` auto-detects in order: CUDA → MLX → XLA/CPU → binary.
Unavailable backends fall back gracefully with a warning.

## Run

```bash
mix deps.get

mix run -e "TernaryAccel.run()"                    # auto-detect
mix run -e "TernaryAccel.run(backend: :mlx)"       # force Apple MLX
mix run -e "TernaryAccel.run(backend: :cuda)"      # force CUDA
```

### Enabling CUDA

EXLA ships CPU-only by default. Rebuild XLA with CUDA support first:

```bash
XLA_TARGET=cuda120 mix deps.get && mix compile   # pick your CUDA version tag
mix run -e "TernaryAccel.run(backend: :cuda)"
```

## Example output (Apple M-series, MLX/Metal)

```
=== TernaryAccel — balanced ternary on Nx ===

Backend: mlx  (Apple Silicon Metal (MLX/EMLX))
GEMM shape: {256, 4096} × {4096, 2048}, 20 iterations

Absmean γ (shared scale):        0.015952
Packed weight sparsity:          31.03% zeros
Weight memory: 33554432 B (f32) → 8388608 B (u8 codes), 4× smaller

=== Accuracy vs FP32 reference ===
Cosine similarity of outputs:    0.887

=== Benchmark (20 iterations, incl. device sync) ===
Dense  FP32 GEMM: 105.8 us/iter  (~40.58 TFLOP/s)
Ternary u8  GEMM: 89.3 us/iter  (~48.1 TFLOP/s)

Per-GEMM weight traffic:
  FP32:            32.0 MB
  u8 codes:        8.0 MB (4× less)
  10→16 packing:   1.6 MB theoretical (.tbin format)
```

The 31% zero fraction matches the theoretical prediction for absmean
quantization of Gaussian weights (see [`concept.md`](../../concept.md) →
"Quantization Theory").

## What to expect

- **Memory/bandwidth**: always 4× less weight traffic (u8 codes); the
  tool's `.tbin` 10-trits-per-16-bit packing gives a further ~1.6×.
- **Compute**: on current GPUs ternary runs through the same matmul
  units as dense, so end-to-end speedups are modest and vary by device.
  The multiplier-free add/sub/skip path requires the purpose-built
  accelerator described in
  [`details/12-custom-ternary-accelerator-design.md`](../../details/12-custom-ternary-accelerator-design.md).

## Tests

```bash
mix test    # exercises the kernels on the CPU fallback; runs anywhere
```
