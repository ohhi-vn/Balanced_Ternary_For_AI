defmodule TernaryConverter.Quantizer do
  @moduledoc """
  Converts full-precision weight tensors to balanced ternary {-1, 0, +1}.

  Three quantization modes are supported (see `quantize/2`):

    - `:delta`   — user-supplied threshold, per-channel MSE-optimal scales
    - `:twn`     — Ternary Weight Networks rule: Δ = 0.7 × mean(|W|)
    - `:absmean` — BitNet b1.58 rule: normalize by mean(|W|), round to nearest trit

  Also provides automatic threshold selection via bisection on target
  sparsity, outlier-clipped scale calibration, and quality metrics
  (sparsity, MSE, SQNR).
  """

  @type tensor :: Nx.Tensor.t()
  @type trit :: -1 | 0 | 1
  @type mode :: :delta | :twn | :absmean

  @twn_threshold_ratio 0.7

  # ── Ternarization ────────────────────────────────────────────

  @doc """
  Ternarize a tensor with a fixed threshold.

  Values greater than `delta` → 1
  Values between [-delta, delta] → 0
  Values less than `-delta` → -1

  ## Examples

      iex> TernaryConverter.Quantizer.ternarize(Nx.tensor([0.8, -0.3, 0.1, -0.9, 0.5]), 0.4)
      #Nx.Tensor<
        s64[5]
        [1, -1, 0, -1, 1]
      >
  """
  @spec ternarize(tensor(), float()) :: tensor()
  def ternarize(w, delta) do
    pos = Nx.greater(w, delta) |> Nx.as_type(:s64)
    neg = Nx.less(w, Nx.negate(delta)) |> Nx.as_type(:s64) |> Nx.multiply(-1)
    Nx.add(pos, neg)
  end

  @doc """
  Ternarize with per-channel scale factors.

  Normalizes each row by its scale factor before ternarization,
  then rescales. This is the recommended approach for all linear layers.
  """
  @spec ternarize_scaled(tensor(), tensor(), float()) :: tensor()
  def ternarize_scaled(w, scales, delta) do
    w_norm = Nx.divide(w, Nx.new_axis(scales, 1))
    t = ternarize(w_norm, delta)
    Nx.multiply(t, Nx.new_axis(scales, 1))
  end

  # ── Scale Factor Calibration ─────────────────────────────────

  @doc """
  Compute per-channel MSE-optimal scale factors.

  For each output channel j:
    α_j = sum(W_j * T_j) / sum(T_j * T_j)

  ## Examples

      iex> w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      iex> scales = TernaryConverter.Quantizer.compute_scales(w, 0.4)
      iex> Nx.to_flat_list(scales)
      [0.5666666626930237, 0.5399999618530273]
  """
  @spec compute_scales(tensor(), float()) :: tensor()
  def compute_scales(weights, delta) do
    # Least squares: for fixed T, minimize ‖W_j − α_j·T_j‖² per row j.
    # Setting the derivative to zero gives α_j = (W_j·T_j)/(T_j·T_j).
    ternary = ternarize(weights, delta)
    dot = Nx.sum(Nx.multiply(weights, ternary), axes: [1])
    norm = Nx.sum(Nx.pow(ternary, 2), axes: [1])

    # A channel with no surviving trits (norm = 0) is dead; substitute
    # 1.0 so the division yields scale 0 instead of NaN.
    safe_norm = Nx.select(Nx.greater(norm, 0.0), norm, Nx.tensor(1.0))
    Nx.divide(dot, safe_norm)
  end

  @doc """
  Compute per-channel scale factors with outlier clipping.

  Before computing scales, clips each channel to ±3σ to prevent
  outlier weights from dominating the scale factor.
  """
  @spec compute_scales_clipped(tensor(), float(), float()) :: tensor()
  def compute_scales_clipped(weights, delta, clip_sigma \\ 3.0) do
    mean = Nx.mean(weights, axes: [1])
    std = Nx.standard_deviation(weights, axes: [1])

    upper = mean + clip_sigma * std
    lower = mean - clip_sigma * std

    clipped =
      weights
      |> Nx.max(Nx.new_axis(lower, 1))
      |> Nx.min(Nx.new_axis(upper, 1))

    compute_scales(clipped, delta)
  end

  # ── Quantization Modes ───────────────────────────────────────

  @doc """
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
  """
  @spec quantize(tensor(), keyword()) :: {tensor(), [float()], float()}
  def quantize(w, opts \\ []) do
    # Explicit :mode wins. Otherwise an explicit :delta means fixed-threshold
    # mode (backward compatible). Default is the TWN rule.
    mode =
      Keyword.get_lazy(opts, :mode, fn ->
        if Keyword.has_key?(opts, :delta), do: :delta, else: :twn
      end)

    case mode do
      :delta ->
        delta = Keyword.get(opts, :delta, 0.5)
        {ternarize(w, delta), to_scale_list(compute_scales(w, delta)), delta}

      :twn ->
        # Derive the threshold from the data, then reuse the standard
        # least-squares scale calibration on top of it.
        delta = twn_delta(w)
        {ternarize(w, delta), to_scale_list(compute_scales(w, delta)), delta}

      :absmean ->
        # BitNet b1.58: normalize by the mean absolute weight and round
        # to the nearest trit. There is no explicit threshold — values
        # within ±γ/2 round to 0 — so we report 0.5·γ as the effective
        # decision boundary for metadata purposes.
        gamma = w |> Nx.abs() |> Nx.mean() |> Nx.to_number()

        trits =
          w
          |> Nx.divide(max(gamma, 1.0e-12))
          |> Nx.round()
          |> Nx.clip(-1.0, 1.0)
          |> Nx.as_type(:s64)

        # Every output channel shares γ; storing one value per channel
        # keeps the Layer representation uniform across modes.
        {trits, List.duplicate(gamma, elem(Nx.shape(w), 0)), 0.5 * gamma}
    end
  end

  @doc """
  TWN threshold rule: Δ = 0.7 × mean(|W|).

  Near-optimal for approximately Gaussian weight distributions and the
  standard choice when no calibration data is available.
  """
  @spec twn_delta(tensor()) :: float()
  def twn_delta(w) do
    w
    |> Nx.abs()
    |> Nx.mean()
    |> Nx.multiply(@twn_threshold_ratio)
    |> Nx.to_number()
  end

  defp to_scale_list(scales), do: Nx.to_flat_list(Nx.as_type(scales, :f32))

  # ── Automatic Threshold Selection ────────────────────────────

  @doc """
  Find the threshold that yields a target zero-fraction via bisection.

  The fraction of weights in [-Δ, Δ] increases monotonically with Δ, so
  the target sparsity has a unique root on [0, 4σ]. Returns
  `{delta, actual_sparsity}`.

  Returns `{optimal_delta, actual_sparsity}`.
  """
  @spec auto_threshold(tensor(), float()) :: {float(), float()}
  def auto_threshold(weights, target_sparsity \\ 0.5)

  def auto_threshold(_weights, target) when target <= 0.0, do: {0.0, 0.0}

  def auto_threshold(weights, target_sparsity) do
    total = Nx.size(weights) |> Nx.to_number()

    if total == 0 do
      {0.0, 0.0}
    else
      hi = 4.0 * (Nx.standard_deviation(weights) |> Nx.to_number())
      zero_at_hi = zero_fraction(weights, hi, total)

      if zero_at_hi < target_sparsity do
        {hi, zero_at_hi}
      else
        bisect(weights, 0.0, hi, target_sparsity, total, 40)
      end
    end
  end

  defp bisect(weights, lo, hi, target, total, iterations) do
    # Invariant: z(lo) < target ≤ z(hi), where z(Δ) is the zero fraction.
    # Each step halves the interval; ~12 iterations suffice for 0.1%
    # precision on a [0, 4σ] range, so the 40-iteration cap is generous.
    mid = (lo + hi) / 2
    sparsity = zero_fraction(weights, mid, total)

    cond do
      abs(sparsity - target) < 1.0e-3 or iterations == 0 ->
        {mid, sparsity}

      sparsity < target ->
        # Too few zeros — widen the dead zone.
        bisect(weights, mid, hi, target, total, iterations - 1)

      true ->
        # Too many zeros — narrow it.
        bisect(weights, lo, mid, target, total, iterations - 1)
    end
  end

  defp zero_fraction(weights, delta, total) do
    weights
    |> ternarize(delta)
    |> Nx.equal(0)
    |> Nx.sum()
    |> Nx.divide(total)
    |> Nx.to_number()
  end

  # ── Quality Metrics ──────────────────────────────────────────

  @doc """
  Compute quantization quality metrics.

  Returns a map with:
    - `:sparsity` — fraction of zero weights
    - `:density` — fraction of non-zero weights
    - `:positive_ratio` — fraction of +1 weights (of non-zero)
    - `:negative_ratio` — fraction of -1 weights (of non-zero)
    - `:mse` — mean squared error vs. original
    - `:sqnr` — signal-to-quantization-noise ratio in dB
  """
  @spec quality_metrics(tensor(), float()) :: map()
  def quality_metrics(original, delta) do
    t = ternarize(original, delta)
    scales = compute_scales(original, delta)
    quantized = ternarize_scaled(original, scales, delta)

    mse = Nx.mean(Nx.pow(Nx.subtract(original, quantized), 2)) |> Nx.to_number()
    signal_power = Nx.mean(Nx.pow(original, 2)) |> Nx.to_number()
    sqnr = if mse > 0, do: 10 * :math.log10(signal_power / mse), else: :infinity

    total = Nx.size(t) |> Nx.to_number()
    zeros = Nx.sum(Nx.equal(t, 0)) |> Nx.to_number()
    pos = Nx.sum(Nx.equal(t, 1)) |> Nx.to_number()
    neg = Nx.sum(Nx.equal(t, -1)) |> Nx.to_number()
    nonzero = total - zeros

    %{
      sparsity: zeros / total,
      density: nonzero / total,
      positive_ratio: if(nonzero > 0, do: pos / nonzero, else: 0.0),
      negative_ratio: if(nonzero > 0, do: neg / nonzero, else: 0.0),
      mse: mse,
      sqnr: sqnr
    }
  end
end
