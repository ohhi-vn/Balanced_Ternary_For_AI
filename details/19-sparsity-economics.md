# 19. Sparsity Economics

Sparsity is not a side effect of ternary quantization — it is the economic
engine. This chapter models where zeros come from, what they cost, and what
they save.

## 19.1 Where Zeros Come From

Three independent sources push trained weights toward zero:

1. **Natural sparsity of trained networks.** Large fractions of weights in
   trained transformers/CNNs are near zero. A binary {−1,+1} network must
   round these to ±1 and pay full price for them; a ternary network maps
   them to exactly 0.

2. **Quantization thresholds.** Every ternarizer — TWN (`Δ = 0.7·mean|W|`),
   absmean (round to nearest trit), or explicit Δ — assigns the central mass
   of the weight distribution to 0. For Gaussian weights this alone yields
   31–50% zeros depending on rule
   ([Quantization Theory](20-quantization-theory.md)).

3. **Explicit sparsity regularization during training** (see
   [Training](03-training.md)): L1 penalties or target-sparsity losses can
   push 75%+ of weights below any threshold, at a small accuracy cost.

Source 2 is free; source 3 buys more savings deliberately. The key insight:

> **The zero trit is simultaneously a quantization level and a pruning
> decision.** Ternary fuses two traditionally separate pipelines —
> quantization and pruning — into one step with no extra tooling.

## 19.2 Accuracy vs Sparsity

Empirical pattern for PTQ conversion of transformer LMs (perplexity delta on
WikiText-style evaluation):

| Sparsity | PPL impact | Compute reduction | Verdict |
|----------|------------|-------------------|---------|
| 0%       | baseline   | 1×                | pointless — no zeros |
| ~31%     | ≈ +0.5     | ~1.45×            | absmean default; nearly free |
| ~42%     | +0.5–1.0   | ~1.7×             | TWN default |
| 50%      | +1.1       | 2×                | sweet spot begins |
| 75%      | +4.0       | 4×                | needs QAT / distillation |
| 90%      | +8.5       | 10×               | research regime |

Reading the table: **the first half of the zeros are cheap, the second half
are expensive.** Production deployments target 40–75%; the shipped tool's
default modes land in the cheap zone without any retraining
([Model Conversion Guide](13-model-conversion-guide.md)).

## 19.3 The Savings Model

For a model with `P` parameters and sparsity `s`:

```
stored_weights    = s·P omitted (with sparse formats) or packed densely
effective_MACs    = (1 − s)·P per token
compute_speedup   ≈ 1/(1 − s)          (GEMM only)
memory_traffic    ∝ format             (dense-packed: unchanged; sparse: ∝ (1−s))
energy_per_token  ≈ activation_energy + (1 − s)·P·e_add
```

### Worked Example: 1B-Parameter Model, 20 tokens/s Decode

| | Dense FP32 | Ternary @ 42% zeros |
|---|-----------|---------------------|
| GEMM adds/token | 1.0 × 10⁹ | 0.58 × 10⁹ |
| Add energy (@ ~0.05 pJ effective) | — | ~29 mJ/s = 29 mW |

The compute term shrinks proportionally to density while the memory term —
usually dominant — shrinks via storage compression
([Memory Bandwidth Analysis](21-memory-bandwidth-analysis.md)).

## 19.4 Why Zeros Must Be *Free* in Hardware

A zero only saves energy if the hardware can exploit it. Requirements:

| Requirement | Mechanism | Where specified |
|-------------|-----------|-----------------|
| Detect zero instantly | differential encoding: `A=B=0` → single NOR gate | [Differential Encoding](09-differential-encoding.md) |
| Gate the lane | clock gating of PE adder input | [Hardware Architecture](05-hardware-architecture.md) |
| Skip memory traffic | sparse formats omit zeros above breakeven | [Storage Format](04-storage-format.md) |
| Keep timing predictable | skipping is data-dependent → use worst-case-throughput design or accept variable latency per tile | [ASIC Design](12-custom-ternary-accelerator-design.md) |

Without these, a software-only skip costs a branch that is often as
expensive as the add it avoids — sparsity pays only in hardware designed
for it, or in bandwidth saved by compressed storage.

## 19.5 Sparse Storage Formats

Dense packing stores every trit (10 per 16-bit word). When sparsity is very
high, storing only non-zeros wins:

| Format | Bits per weight | Breakeven sparsity | Notes |
|--------|-----------------|--------------------|-------|
| Dense packed (10→16) | 1.6 | — | constant latency, simplest decoder |
| Index+sign entries | 16 bits per nonzero | > ~88% | implemented as `Packer.pack_sparse/1` |
| Block-sparse (32×32 blocks) | metadata + values | > ~75% block-level | preserves locality for tiled GEMM |

Breakeven logic: an index+sign entry costs 16 bits; dense packing costs
1.6 bits/weight. Storing indices beats dense only when fewer than ~10% of
weights are nonzero. Most production models sit below that line, which is
why **dense packed remains the default** and sparse encoding is an option,
not the primary path.

## 19.6 Measuring Sparsity with the Toolchain

```bash
# Convert and report per-layer sparsity
mix run -e 'TernaryConverter.CLI.main(["convert",
  "--input", "model.safetensors", "--arch", "llama", "--mode", "twn"])'
```

```elixir
# Programmatic metrics for one layer
{T, _scales, _delta} =
  TernaryConverter.Quantizer.quantize(w, mode: :absmean)

zeros = Nx.sum(Nx.equal(T, 0)) |> Nx.divide(Nx.size(T)) |> Nx.to_number()

# Full-model statistics including per-layer breakdown
{:ok, layers, meta} = TernaryConverter.load("model.tbin")
stats = TernaryConverter.Stats.compute(layers)
stats.overall_sparsity   # e.g. 0.42
```

Per-layer sparsity matters more than the average: embedding-adjacent and
down-projection layers often differ by 10+ points, and the *minimum*
sparsity layer bounds the speedup of uniform-tile hardware
([Sensitivity Analysis](02-weight-quantization.md)).

## 19.7 Structured vs Unstructured Zeros

| Property | Unstructured (per-weight) | Structured (blocks/channels) |
|----------|---------------------------|------------------------------|
| Best accuracy at same s | ✔ better | slightly worse |
| Hardware utilization | poor unless lanes gated individually | good — whole tiles skip |
| Memory traffic saving | needs fine-grained sparse formats | simple contiguous skips |
| Fit with systolic arrays | weak | strong |

Ternary's zero trit is unstructured by construction, but the differential
encoding makes per-lane gating cheap enough ([09](09-differential-encoding.md))
that unstructured sparsity remains practical. Block-structured training
objectives remain an open optimization documented in
[Training](03-training.md).

## 19.8 Summary

- Zeros arrive from both quantization (free) and training (deliberate).
- First-half zeros cost almost no accuracy: 30–50% sparsity is the
  production sweet spot.
- Savings require hardware cooperation: detect (NOR), gate (clock), omit
  (sparse format).
- Measure per layer, not on average; convert with `mode: :absmean` or
  `mode: :twn` to get the free zone automatically.

---

## 19.9 Knowledge Check

### Concept Questions

**Q1.** Why is the *first* half of zeros nearly accuracy-free while the
second half is expensive? Answer in terms of the weight distribution.

**Q2.** Index+sign sparse encoding stores each nonzero in 16 bits. Derive
the breakeven sparsity above which it beats dense 10→16 packing, and
explain why real models rarely cross it.

**Q3.** A model averages 50% sparsity, but one layer is only 20% sparse.
Why does the *minimum* layer matter for uniform-tile hardware?

### Calculations

**Q4.** For a Gaussian with σ = 0.05, TWN sets Δ = 0.7·mean|W|. Estimate
Δ and the expected zero fraction.

**Q5.** A 2B-parameter model at 45% sparsity: how many effective adds per
token? If each add costs ~0.05 pJ effective, what is GEMM power at 30
tokens/s?

### Hands-On Lab

Measure the sparsity/accuracy-of-reconstruction tradeoff across modes:

```bash
cd tools/ternary_converter && mix deps.get
mix run -e 'TernaryConverter.CLI.main(["convert", "--output", "s42.tbin", "--sparsity", "0.42"])'
mix run -e 'TernaryConverter.CLI.main(["info", "--model", "s42.tbin"])'
```

Then compare modes on identical weights:

```elixir
w = Nx.Random.normal(Nx.Random.key(1), 0.0, 0.02, shape: {512, 512}) |> elem(0)

for mode <- [:delta, :twn, :absmean] do
  {t, _scales, delta} = TernaryConverter.Quantizer.quantize(w, mode: mode)
  zeros = Nx.sum(Nx.equal(t, 0)) |> Nx.divide(Nx.size(t)) |> Nx.to_number()
  IO.puts("#{mode}: zeros=#{Float.round(zeros * 100, 1)}% delta=#{Float.round(delta, 5)}")
end
```

Exercises:
1. Confirm absmean ≈ 31% and twn ≈ 42% on Gaussian weights.
2. Replace the weights with a Laplace distribution (`Nx.random_normal`
   twice, sign-random) and re-measure — which rule shifts most, and why?
   (Hint: heavier tails → more mass beyond Δ.)

### Answer Key

<details>
<summary>Click to reveal</summary>

**A1.** Weights are bell-shaped: the central mass within ±Δ is dense with
near-zero values that contribute little to outputs — zeroing them barely
changes activations. Beyond ~50%, you start deleting weights in the
distribution's shoulders/tails that carry proportionally more signal, so
each additional point of sparsity costs more perplexity.

**A2.** Dense costs 16 bits per 10 weights = 1.6 bits/weight; index+sign
costs 16 bits per nonzero = 16·(1−s) bits/weight. Breakeven:
`16(1−s) = 1.6 ⟹ s = 90%`. Real PTQ models sit at 30–60%, so dense
packing wins; only extremely pruned models justify indices.

**A3.** Tiles iterate over all lanes uniformly; a layer below the assumed
sparity forces the array to run dense for its tiles, setting the latency
floor. End-to-end speedup is bounded by the least-sparse layer unless
scheduling adapts per tile.

**A4.** mean|W| ≈ 0.798σ = 0.0399; Δ ≈ 0.7 × 0.0399 ≈ 0.028.
Zero fraction ≈ 42% (from §20.6 of [Quantization Theory](20-quantization-theory.md)).

**A5.** Effective adds/token = 2e9 × 0.55 = 1.1e9. At 30 tokens/s:
`1.1e9 × 30 × 0.05 pJ = 1.65e12 × 0.05 pJ = 82.5 µJ/s ≈ 83 nW` for GEMM
adds alone — the point being that compute energy is negligible next to
memory movement ([21](21-memory-bandwidth-analysis.md)).

</details>
