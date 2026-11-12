# `BalancedTernaryEngine.Quantizer`

Converts full-precision weight tensors to balanced ternary {-1, 0, +1}.

Three quantization modes are supported (see `quantize/2`):

  - `:delta`   — user-supplied threshold, per-channel MSE-optimal scales
  - `:twn`     — Ternary Weight Networks rule: Δ = 0.7 × mean(|W|)
  - `:absmean` — BitNet b1.58 rule: normalize by mean(|W|), round to nearest trit

Also provides automatic threshold selection via bisection on target
sparsity, outlier-clipped scale calibration, and quality metrics
(sparsity, MSE, SQNR).

# `mode`

```elixir
@type mode() :: :delta | :twn | :absmean
```

# `tensor`

```elixir
@type tensor() :: Nx.Tensor.t()
```

# `trit`

```elixir
@type trit() :: -1 | 0 | 1
```

# `auto_threshold`

```elixir
@spec auto_threshold(tensor(), float()) :: {float(), float()}
```

Find the threshold that yields a target zero-fraction via bisection.

The fraction of weights in [-Δ, Δ] increases monotonically with Δ, so
the target sparsity has a unique root on [0, 4σ]. Returns
`{delta, actual_sparsity}`.

Returns `{optimal_delta, actual_sparsity}`.

# `compute_scales`

```elixir
@spec compute_scales(tensor(), float()) :: tensor()
```

Compute per-channel MSE-optimal scale factors.

For each output channel j:
  α_j = sum(W_j * T_j) / sum(T_j * T_j)

## Examples

    iex> w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
    iex> scales = BalancedTernaryEngine.Quantizer.compute_scales(w, 0.4)
    iex> Nx.to_flat_list(scales)
    [0.5666666626930237, 0.5399999618530273]

# `compute_scales_clipped`

```elixir
@spec compute_scales_clipped(tensor(), float(), float()) :: tensor()
```

Compute per-channel scale factors with outlier clipping.

Before computing scales, clips each channel to ±3σ to prevent
outlier weights from dominating the scale factor.

# `quality_metrics`

```elixir
@spec quality_metrics(tensor(), float()) :: map()
```

Compute quantization quality metrics.

Returns a map with:
  - `:sparsity` — fraction of zero weights
  - `:density` — fraction of non-zero weights
  - `:positive_ratio` — fraction of +1 weights (of non-zero)
  - `:negative_ratio` — fraction of -1 weights (of non-zero)
  - `:mse` — mean squared error vs. original
  - `:sqnr` — signal-to-quantization-noise ratio in dB

# `quantize`

```elixir
@spec quantize(tensor(), keyword()) :: {tensor(), [float()], float()}
```

Quantize a weight tensor to trits plus per-channel scales.

Returns `{trits, scales, delta}` where:

  - `trits` — s64 tensor of {-1, 0, +1}, same shape as `w`
  - `scales` — flat list of per-channel f32 scale factors
  - `delta` — the effective decision threshold used

## Modes

  - `:delta` — threshold at the given `:delta`, per-channel MSE-optimal
    scales αⱼ = (Wⱼ·Tⱼ)/(Tⱼ·Tⱼ)
  - `:twn` — Ternary Weight Networks rule: Δ = 0.7 × mean(|W|), then
    MSE-optimal per-channel scales (Li & Liu, 2016)
  - `:absmean` — BitNet b1.58 rule: T = roundclip(W / mean(|W|), -1, +1),
    single shared scale γ = mean(|W|) for every channel (Ma et al., 2024)

## Options

  - `:mode` — `:delta` | `:twn` | `:absmean` (default: `:twn`)
  - `:delta` — threshold used when mode is `:delta`

# `ternarize`

```elixir
@spec ternarize(tensor(), float()) :: tensor()
```

Ternarize a tensor with a fixed threshold.

Values greater than `delta` → 1
Values between [-delta, delta] → 0
Values less than `-delta` → -1

## Examples

    iex> BalancedTernaryEngine.Quantizer.ternarize(Nx.tensor([0.8, -0.3, 0.1, -0.9, 0.5]), 0.4)
    #Nx.Tensor<
      s64[5]
      [1, -1, 0, -1, 1]
    >

# `ternarize_scaled`

```elixir
@spec ternarize_scaled(tensor(), tensor(), float()) :: tensor()
```

Ternarize with per-channel scale factors.

Normalizes each row by its scale factor before ternarization,
then rescales. This is the recommended approach for all linear layers.

# `twn_delta`

```elixir
@spec twn_delta(tensor()) :: float()
```

TWN threshold rule: Δ = 0.7 × mean(|W|).

Near-optimal for approximately Gaussian weight distributions and the
standard choice when no calibration data is available.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
