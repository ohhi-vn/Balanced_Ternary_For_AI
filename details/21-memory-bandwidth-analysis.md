# 21. Memory Bandwidth: The Real Bottleneck

Modern LLM inference is not compute-bound — it is **memory-bandwidth-bound**.
This chapter works through the arithmetic-intensity argument, sizes the
on-chip storage that ternary enables, and breaks down where inference energy
actually goes. Companion material: throughput modeling in
[LLM Inference Engine](06-llm-inference-engine.md), cost scaling in
[Feasibility](14-ternary-llm-feasibility.md).

## 21.1 Arithmetic Intensity of Decoding

Generating one token requires touching every weight exactly once (per
layer, per token). With `P` parameters:

```
FLOPs  ≈ 2P            (one multiply + one add per weight)
Bytes  = P × bits/8    (weights read once)
AI     = FLOPs / Bytes
       = 16 / bits      (FLOP per byte — independent of model size!)
```

| Format | Bits | Arithmetic intensity | Regime at batch 1 |
|--------|------|---------------------|--------------------|
| FP32   | 32   | 0.5 FLOP/B | deeply memory-bound |
| INT8   | 8    | 2 FLOP/B   | memory-bound |
| INT4   | 4    | 4 FLOP/B   | memory-bound |
| **Ternary** | **1.585** | **~10 FLOP/B** | approaching compute floor |

A GPU with ~100 TFLOP/s and ~1 TB/s needs AI ≈ 100 FLOP/B to saturate
compute. Every quantized format fails that test at batch 1 — but the gap
*defines the win*: time-per-token is set by bytes moved, so cutting bits per
weight cuts latency proportionally.

## 21.2 Decode-Phase Bandwidth Arithmetic

Tokens/s at batch 1 (memory-bound):

```
tokens_per_second ≤ bandwidth / bytes_per_token
bytes_per_token   ≈ P × bits/8  (+ KV cache reads)
```

For a 1B-parameter model at 20 tokens/s:

| Format | Weights size | Bandwidth needed |
|--------|--------------|------------------|
| FP32   | 4 GB         | 80 GB/s          |
| INT8   | 1 GB         | 20 GB/s          |
| INT4   | 0.5 GB       | 10 GB/s          |
| Ternary (dense packed) | ~200 MB | **4 GB/s** |

4 GB/s fits comfortably in LPDDR-class memory; more importantly, 200 MB can
live entirely **on-chip**, reducing effective bandwidth demand to zero.

### KV Cache Correction

Attention reads the KV cache too: `2 × layers × kv_heads × head_dim ×
seq_len × bytes`. For long contexts this term can rival weight traffic —
which is why GQA-aware caching matters ([LLM Engine](06-llm-inference-engine.md))
and why ternarizing weights does not by itself fix attention-heavy phases.

## 21.3 On-Chip SRAM Fit

The qualitative jump: ternary moves weights from DRAM to SRAM.

| Format | 1B parameters | Fits in on-chip SRAM? |
|--------|---------------|------------------------|
| FP32   | 4 GB          | impossible (dies are ~mm²) |
| INT8   | 1 GB          | no |
| INT4   | 512 MB        | no |
| Ternary | **~200 MB**  | borderline; with sparse encoding / MoE experts, yes for sub-1B |

Why SRAM matters (order-of-magnitude, same node):

| Metric | DRAM read | SRAM read |
|--------|-----------|-----------|
| Energy per access | ~10–100 pJ | ~0.1–1 pJ |
| Latency | ~100 ns | ~1 ns |
| Bandwidth | shared bus | per-tile parallel |

The [ASIC design](12-custom-ternary-accelerator-design.md) allocates 16 MB
SRAM tiles feeding a 256×256 PE array precisely because a 1B ternary model
divides into on-chip-resident shards.

## 21.4 Energy Breakdown

Share of inference energy for a memory-bound decode workload:

| Component | FP32 GPU | Ternary ASIC | What changed |
|-----------|----------|--------------|--------------|
| Weight fetch | ~60% | ~0% | SRAM-resident; no DRAM stream |
| Compute (MAC) | ~25% | ~8% | multiplier removed ([18](18-multiplication-as-selection.md)) |
| Activation movement | ~10% | ~10% | unchanged (INT8 activations) |
| Softmax/norm/control | ~5% | ~2% | simpler control; FP16 units retained |

Two structural facts fall out:

1. **Optimizing compute alone cannot help much** when 60% of energy is
   memory stalls — which is why bit-width reduction (storage) beats ALU
   tricks.
2. After ternarization the remaining budget is dominated by activation
   movement — the next optimization frontier (activation quantization,
   fusion), out of scope here.

## 21.5 Batch Changes the Picture

Arithmetic intensity scales with batch size B (weights are reused across
batch elements):

```
AI(B) ≈ B × 16 / bits
```

| Batch | FP32 AI | Ternary AI |
|-------|---------|------------|
| 1 (decode) | 0.5 | ~10 |
| 32 | 16 | ~320 |
| 256 (prefill-like) | 128 | ~2560 |

At large batches even FP32 approaches the compute roofline, and the ternary
advantage shifts from "less waiting" to "cheaper arithmetic"
([18](18-multiplication-as-selection.md)). Practical consequence:

- **Decode (B=1)** → ternary wins big via bandwidth.
- **Prefill (large B)** → both formats become compute-limited; ternary wins
  via simpler PEs, not bandwidth.

## 21.6 Measuring on Real Hardware Today

The [`examples/ternary_accel`](../../examples/ternary_accel/) demo reports
exactly this traffic accounting on live GPUs. Sample from Apple MLX/Metal:

```
GEMM shape: {256, 4096} × {4096, 2048}, 20 iterations
Weight memory: 33554432 B (f32) → 8388608 B (u8 codes), 4× smaller

Per-GEMM weight traffic:
  FP32:            32.0 MB
  u8 codes:        8.0 MB (4× less)
  10→16 packing:   1.6 MB theoretical (.tbin format)
```

Notes for interpreting such measurements:

- Byte codes (`u8`) demonstrate the pipeline portably; the `.tbin` packing
  reaches the theoretical ~1.6-bit density.
- On current GPUs ternary still executes as a matmul after on-device decode,
  so *compute* speedups are modest; the guaranteed gain is the 4–20×
  traffic reduction, which becomes decisive once models exceed cache.

## 21.7 Summary Checklist

When evaluating any low-precision format for LLM inference, compute:

- [ ] bytes/token = P·bits/8 (+KV) → tokens/s ceiling at your bandwidth
- [ ] does the working set fit on-chip? (ternary uniquely plausible)
- [ ] batch regime: memory-bound (decode) or compute-bound (prefill)?
- [ ] energy split before/after — attack the largest remaining term

Ternary's case rests on attacking all four simultaneously: fewest bytes,
smallest on-chip footprint, adequate compute simplicity, and an energy
profile dominated by terms it has already minimized.

---

## 21.8 Knowledge Check

### Concept Questions

**Q1.** Why is arithmetic intensity at batch 1 *independent of model
size*? What does that imply for "bigger models need faster GPUs"?

**Q2.** A vendor advertises 400 TFLOP/s and 2 TB/s. What AI must a
workload have to be compute-bound? Can any weight-only quantized LLM
decode reach that at batch 1?

**Q3.** Why does the KV cache undermine pure weight-size arguments at
long context, and which architectural feature mitigates it?

### Calculations

**Q4.** 3B-parameter model, ternary dense-packed, target 25 tokens/s at
batch 1. Compute required bandwidth; repeat for INT4.

**Q5.** At what batch size does a ternary model (AI ≈ 10·B) cross the
compute roofline of Q2's GPU? What phase of inference does that
correspond to?

**Q6.** Energy: FP32 decode spends 60% of 15 W on weight fetch. Ternary
moves weights to SRAM (~1% of budget). How many watts are freed, and
what could bound the new bottleneck?

### Hands-On Lab

Reproduce the traffic analysis on your own hardware:

```bash
cd examples/ternary_accel && mix deps.get
mix run -e 'TernaryAccel.run(backend: :binary)'   # works anywhere
mix run -e 'TernaryAccel.run()'                   # auto-detect CUDA/MLX
```

Exercises:
1. From the demo output, compute measured GB/s for each GEMM variant
   (`bytes_moved / time_per_iter`) and compare with your device's spec
   sheet bandwidth.
2. Scale `rows:`/`cols:` until the u8-codes GEMM exceeds your L2 cache;
   observe whether the ternary-vs-dense ratio improves as you leave the
   cache hierarchy (it should — that is the memory-bound thesis).
3. Modify the demo to add an INT4-style simulation (pack two 4-bit codes
   per byte) and extend the traffic table with its row.

### Answer Key

<details>
<summary>Click to reveal</summary>

**A1.** Decoding touches every weight once per token: FLOPs ≈ 2P and
bytes = P·bits/8 both scale linearly with P, so AI = 16/bits regardless
of P. Implication: bigger models don't become compute-bound at batch 1 —
they just need proportionally more bandwidth, so *bandwidth*, not peak
FLOP/s, predicts tokens/s across the size range.

**A2.** Compute-bound needs AI > TFLOP/s ÷ TB/s = 200 FLOP/B. Weight-only
quantized decode tops out at ~16/bits ≤ 10 FLOP/B for ternary — two
orders short. Only batching or attention/KV-heavy shapes change this.

**A3.** KV reads grow with sequence length and can exceed weight traffic:
`2 · layers · kv_heads · head_dim · seq_len × bytes`. Mitigation: GQA /
MQA (shared KV heads), which cuts KV bytes by the head-sharing factor —
one reason LLaMA-class models pair well with ternary weights.

**A4.** Bytes/token = 3e9 × 1.585/8 ≈ 0.6 GB → 0.6 × 25 ≈ **14.6 GB/s**
(ternary); INT4: 1.5 GB → **37.5 GB/s**. Both feasible in LPDDR, but only
ternary approaches single-channel class budgets.

**A5.** Solve 10·B > 200 ⟹ B > 20. That is prefill / batched-serving
territory — exactly where the win shifts from bandwidth to cheaper PEs
(§21.5).

**A6.** Freed: 0.6 × 15 W = 9 W. New bottleneck candidates: activation
movement between tiles and the FP16 softmax/norm units — i.e., the
remaining non-ternary terms of §21.4's table.

</details>
