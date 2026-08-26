# 22. History and Long-Term Vision

Balanced ternary is not a new idea — it is an old idea that finally found
its workload. This chapter records where the idea came from, why it lost the
first time, what changed, and where a realistic development program leads.

## 22.1 The Setun Precedent (1958)

The most famous ternary computer was built at Moscow State University:

- **Setun** (Николай Брусенцов / Nikolai Brusentsov, 1958): ~50 machines
  built; used **trits** natively with 18-trit words.
- Followed by **Setun-70** (1970), which pushed the architecture further
  (stack orientation, structured programming support).

Why ternary appealed then — and still does mathematically:

| Property | Binary | Balanced ternary |
|----------|--------|------------------|
| Radix economy (cost ≈ r·log_r N) | baseline | near-optimal (r = e ≈ 2.718) |
| Sign representation | separate sign bit | native (leading trit) |
| Negation | two's complement trick | single trit flip |
| Rounding in division/truncation | asymmetric bias | symmetric (round-toward-zero natural) |
| Comparison to zero | dedicated test | inspect sign trit |

## 22.2 Why Binary Won Anyway

Binary's victory was not mathematical — it was manufacturing economics:

1. **Switch simplicity.** A binary gate needs one stable threshold region;
   three-level storage/signalling demands tighter margins for negligible
   density gain in wires.
2. **Noise immunity scales with levels squared** — binary maximizes margin
   per level.
3. **Ecosystem lock-in.** Once logic families, compilers, and instruction
   sets standardized on bits, no workload justified retooling.

The lesson recorded here for modern efforts: **do not fight binary on its
turf.** Ternary wins only where the third state carries information that
binary must simulate expensively — exactly the sparsity argument of this
project ([concept](../concept.md)).

## 22.3 What AI Changes

Four properties of neural network workloads invert Setun-era calculus:

| Property | Consequence for ternary |
|----------|--------------------------|
| **Error-tolerant** | quantization noise (SQNR ~9–10 dB, [20](20-quantization-theory.md)) is absorbable; exactness unnecessary |
| **Memory-bandwidth-bound** | 20× weight compression attacks the binding constraint directly ([21](21-memory-bandwidth-analysis.md)) |
| **Multiplication-heavy** | selection replaces multiplication; PE cost collapses ([18](18-multiplication-as-selection.md)) |
| **Naturally sparse** | the third state encodes "unimportant" natively — pruning becomes free ([19](19-sparsity-economics.md)) |

No prior workload offered all four at once. Scientific computing demanded
exactness; databases were I/O-bound but not multiplier-bound; DSP was
fixed-function. Neural networks are the first mainstream workload whose
*statistical* nature makes a three-valued representation strictly cheaper
per unit of useful work.

Supporting evidence that this is more than theory: the BitNet b1.58 family
demonstrates trained ternary LLMs at billion-parameter scale, and INT4/FP8
products show the market already pays for aggressive quantization — ternary
is the next step down the same curve, not a detour from it.

## 22.4 Target Markets

Ranked by fit between ternary strengths (storage, energy) and market pain:

| Market | Model size | Power budget | Key metric | Why ternary fits |
|--------|-----------|--------------|------------|-------------------|
| Smartphones | 100M–1B | <3 W | on-device LLM | SRAM-resident weights; battery-driven |
| Edge AI boxes | 100M–7B | <5 W | real-time inference | bandwidth ceiling lifted 20× |
| Drones | 50M–500M | <2 W | weight + power mass | model fits flight-controller SRAM |
| Robotics | 100M–1B | <5 W | latency | deterministic decode-phase latency |
| IoT sensors | 10M–100M | <1 W | cost + power | tiny dies viable (20 mm² ASIC class) |

Common thread: every market is constrained by *memory movement under a
power budget*, never by peak FLOP/s. That is precisely the profile analyzed
in [21](21-memory-bandwidth-analysis.md).

## 22.5 Research Trajectory

The 48-week execution plan lives in [Research Roadmap](08-research-roadmap.md);
the strategic shape:

```
Phase 0: Foundation        → software simulation, literature base      ← tools in this repo
Phase 1: Small Models      → ResNet/BERT PTQ + QAT validation          ← converter ready
Phase 2: Transformer       → GPT-2/LLaMA-1B accuracy targets           ← safetensors path ready
Phase 3: Sparsity          → 75%+ trained sparsity                     ← training recipes needed
Phase 4: HW Simulation     → cycle-accurate simulator
Phase 5: FPGA Prototype    → add/sub/skip datapath in fabric           ← guide in [16]
Phase 6: ASIC Pathfinding  → tape-out specification                    ← guides in [12][17]
```

Exit criteria worth stating explicitly:

- Phase 1→2 gate: PTQ perplexity delta ≤ 2% on BERT-class tasks.
- Phase 2→3 gate: LLaMA-1B within 3% relative PPL of FP16 baseline.
- Phase 4→5 gate: simulated tokens/s ≥ published estimates for equal area.
- Phase 5→6 gate: FPGA measured energy/MAC ≤ INT8 baseline × 0.3.

## 22.6 Honest Risks

| Risk | Assessment | Mitigation tracked in |
|------|------------|------------------------|
| INT4/FP8 keep improving | strong ecosystems; -0.5% accuracy today | compete on energy+SRAM, not bits alone ([10](10-comparison.md)) |
| Software-first BitNet path bypasses custom silicon | plausible near-term outcome | our toolchain targets both (.tbin runs on GPU today) |
| Sparsity gains need hardware to cash in | true by construction ([19](19-sparsity-economics.md)) | FPGA phase validates before ASIC spend |
| Accuracy gap at 75%+ sparsity | real cost, documented | QAT/distillation recipes ([03](03-training.md), [07](07-accuracy-analysis.md)) |

## 22.7 Position Statement

Balanced ternary will likely not replace binary computing — Setun taught
that lesson permanently. The realistic, evidence-backed vision is narrower:

> **A specialized inference substrate**: ternary weights resident in on-chip
> SRAM, processed by multiplier-free arrays, serving the decode-heavy,
> memory-starved, error-tolerant workloads that define edge AI.

Everything in this repository — the quantization theory ([20](20-quantization-theory.md)),
the conversion toolchain ([13](13-model-conversion-guide.md)), the hardware
specifications ([12](12-custom-ternary-accelerator-design.md),
[17](17-asic-implementation-guide.md)) — exists to make that sentence
testable rather than aspirational.

---

## 22.8 Knowledge Check

### Concept Questions

**Q1.** Setun's radix economy argument favors r ≈ e ≈ 2.72, so why did
binary's *manufacturing* argument outweigh it? Answer with one sentence
per factor: switching margin, noise immunity, ecosystem.

**Q2.** Which single property of neural networks do you consider most
decisive for ternary's case, and which historical workload (scientific,
DSP, database) came closest to needing a third state without getting it?

**Q3.** The "software-first BitNet path" risk says GPUs may serve ternary
well enough that custom silicon never ships. What evidence in this repo
argues both sides?

### Calculations

**Q4.** Radix economy: compute r·⌈log_r 1000⌉ for r = 2 and r = 3.
How much does ternary save on digit-storage cost for numbers up to 1000?

**Q5.** From §22.4: an edge box has a 5 W budget and needs a 3B model at
15 tokens/s. Using bytes/token arithmetic from [21](21-memory-bandwidth-analysis.md),
show whether ternary dense-packed fits; what about INT8?

### Hands-On Lab

Turn the vision into a falsifiable experiment:

1. Pick two checkpoints (any small safetensors model).
2. Convert both at `--mode twn` and `--mode absmean`; record per-layer
   sparsity and compression from `info`.
3. Run [`examples/ternary_accel`](../../examples/ternary_accel/) on your
   hardware; record traffic ratios and cosine similarity.
4. Write a one-paragraph verdict: does the measured evidence support or
   contradict the §22.7 position statement? Which claim was weakest?

### Answer Key

<details>
<summary>Click to reveal</summary>

**A1.** Switching margin: binary needs one threshold region vs ternary's
two — simpler gates. Noise immunity scales with level spacing squared;
binary maximizes it per wire. Ecosystem: once logic families, tools, and
ISMs standardized on bits, no workload justified retooling until AI.

**A2.** Defensible pick: memory-bound error tolerance — the combination
lets approximate three-valued storage replace exact binary computation
where bandwidth, not precision, binds. Closest historical near-miss:
DSP/fixed-point signal processing (coefficients often small integers,
multipliers costly) — but it demanded determinism and never produced
naturally sparse coefficient sets.

**A3.** For: the toolchain already serves ternary on GPUs/MLX today with
4–20× traffic wins (`ternary_accel`), weakening the ASIC urgency.
Against: current GPUs still execute dense matmuls after decode — no
lane gating, so sparsity's energy win stays unclaimed until custom
datapaths exist ([18](18-multiplication-as-selection.md), [19](19-sparsity-economics.md)).

**A4.** Binary: 2·⌈log₂1000⌉ = 2·10 = 20. Ternary: 3·⌈log₃1000⌉ =
3·7 = 21... but balanced ternary represents ±1000 natively without a
sign digit; counting signed range (±59049 vs 0..1023) ternary wins on
symmetric ranges: 6 trits cover ±729, needing ~11 bits + sign in binary.
The classic result: cost ratio 3/log₂3 ≈ 1.892 vs binary for unsigned,
and better once signs matter.

**A5.** Ternary: 3e9 × 1.585/8 ≈ 0.6 GB/token → ×15 = **8.9 GB/s**
— feasible under a 5 W envelope with LPDDR5 (~30–60 GB/s). INT8:
3 GB → 44 GB/s sustained — borderline, likely forces DRAM streaming and
breaks the power budget. Verdict: only ternary comfortably fits.

</details>
