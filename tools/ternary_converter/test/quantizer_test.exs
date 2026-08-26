defmodule TernaryConverter.QuantizerTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.Quantizer

  describe "ternarize/2" do
    test "basic ternarization with delta" do
      w = Nx.tensor([0.8, -0.3, 0.1, -0.9, 0.5])
      result = Quantizer.ternarize(w, 0.4)
      # 0.8 > 0.4 → 1, -0.3 > -0.4 → 0, 0.1 ≤ 0.4 → 0, -0.9 < -0.4 → -1, 0.5 > 0.4 → 1
      assert Nx.to_flat_list(result) == [1, 0, 0, -1, 1]
    end

    test "all zeros when delta is very large" do
      w = Nx.tensor([0.1, -0.2, 0.3])
      result = Quantizer.ternarize(w, 1.0)
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "no zeros when delta is zero" do
      w = Nx.tensor([0.1, -0.2, 0.3])
      result = Quantizer.ternarize(w, 0.0)
      assert Nx.to_flat_list(result) == [1, -1, 1]
    end

    test "2D tensor ternarization" do
      w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      result = Quantizer.ternarize(w, 0.4)
      # Row 0: 0.8>0.4→1, -0.3>-0.4→0, 0.1≤0.4→0
      # Row 1: -0.9<-0.4→-1, 0.5>0.4→1, -0.2>-0.4→0
      assert Nx.to_flat_list(result) == [1, 0, 0, -1, 1, 0]
    end

    test "exact boundary values" do
      w = Nx.tensor([0.5, -0.5, 0.0])
      result = Quantizer.ternarize(w, 0.5)
      # 0.5 is NOT > 0.5, so → 0; -0.5 is NOT < -0.5, so → 0; 0.0 → 0
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "just above/below boundary" do
      w = Nx.tensor([0.51, -0.51])
      result = Quantizer.ternarize(w, 0.5)
      assert Nx.to_flat_list(result) == [1, -1]
    end
  end

  describe "compute_scales/2" do
    test "computes per-channel scales" do
      w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      scales = Quantizer.compute_scales(w, 0.4)
      list = Nx.to_flat_list(scales)
      assert length(list) == 2
      assert Enum.all?(list, &(&1 > 0))
    end

    test "handles all-zero channel" do
      w = Nx.tensor([[0.01, 0.02, 0.03], [0.8, -0.3, 0.1]])
      scales = Quantizer.compute_scales(w, 0.4)
      list = Nx.to_flat_list(scales)
      # First channel: all values within [-0.4, 0.4], so all ternarized to 0
      # norm = 0, so scale should be 0.0 (dot product is 0)
      assert Enum.at(list, 0) == 0.0
    end
  end

  describe "quality_metrics/2" do
    test "returns valid metrics" do
      w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      metrics = Quantizer.quality_metrics(w, 0.4)

      assert is_map(metrics)
      assert Map.has_key?(metrics, :sparsity)
      assert Map.has_key?(metrics, :density)
      assert Map.has_key?(metrics, :mse)
      assert Map.has_key?(metrics, :sqnr)

      assert metrics.sparsity >= 0.0 and metrics.sparsity <= 1.0
      assert metrics.density >= 0.0 and metrics.density <= 1.0
      assert metrics.mse >= 0.0
    end
  end

  describe "twn_delta/1" do
    test "is 0.7 times the mean absolute weight" do
      w = Nx.tensor([[1.0, -1.0, 2.0], [-4.0, 0.5, 3.5]])
      expected = w |> Nx.abs() |> Nx.mean() |> Nx.multiply(0.7) |> Nx.to_number()
      assert_in_delta Quantizer.twn_delta(w), expected, 1.0e-6
    end
  end

  describe "auto_threshold/2" do
    test "reaches target sparsity for Gaussian weights" do
      key = Nx.Random.key(7)
      {w, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {64, 128})

      {delta, actual} = Quantizer.auto_threshold(w, 0.5)
      assert delta > 0.0
      assert abs(actual - 0.5) < 0.02
    end

    test "higher target sparsity needs larger delta" do
      key = Nx.Random.key(8)
      {w, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {32, 64})

      {d50, _} = Quantizer.auto_threshold(w, 0.5)
      {d75, _} = Quantizer.auto_threshold(w, 0.75)
      assert d75 > d50
    end

    test "target 0 returns zero threshold" do
      assert Quantizer.auto_threshold(Nx.tensor([1.0, 2.0]), 0.0) == {0.0, 0.0}
    end
  end

  describe "quantize/2 modes" do
    setup do
      key = Nx.Random.key(11)
      # Gaussian weights with a clear structure
      {w, _} = Nx.Random.normal(key, 0.0, 0.05, shape: {16, 64})
      {:ok, w: w}
    end

    test "explicit delta mode uses the given threshold", %{w: w} do
      {trits, scales, delta} = Quantizer.quantize(w, mode: :delta, delta: 0.08)
      assert delta == 0.08
      assert length(scales) == 16
      values = Nx.to_flat_list(trits) |> Enum.uniq()
      assert Enum.all?(values, &(&1 in [-1, 0, 1]))
    end

    test "passing only delta keeps backward-compatible fixed-delta behavior", %{w: w} do
      {trits_a, scales_a, delta_a} = Quantizer.quantize(w, delta: 0.06)
      {trits_b, scales_b, _} = Quantizer.quantize(w, mode: :delta, delta: 0.06)

      assert trits_a == trits_b
      assert scales_a == scales_b
      assert delta_a == 0.06
    end

    test "twn mode derives threshold from mean absolute weight", %{w: w} do
      {_trits, scales, delta} = Quantizer.quantize(w, mode: :twn)
      assert_in_delta delta, Quantizer.twn_delta(w), 1.0e-6
      assert length(scales) == 16
      assert Enum.all?(scales, &(&1 != 0.0))
    end

    test "absmean mode normalizes by mean absolute weight", %{w: w} do
      {trits, scales, _delta} = Quantizer.quantize(w, mode: :absmean)

      gamma = Nx.mean(Nx.abs(w)) |> Nx.to_number()
      # Single shared scale repeated per channel
      assert Enum.uniq(scales) == [gamma]

      values = Nx.to_flat_list(trits)
      assert Enum.all?(values, &(&1 in [-1, 0, 1]))

      # Reconstruction must beat the all-zero baseline (positive SQNR)
      reconstructed = Nx.multiply(trits, gamma)

      mse =
        w
        |> Nx.subtract(reconstructed)
        |> Nx.pow(2)
        |> Nx.mean()
        |> Nx.to_number()

      signal = Nx.mean(Nx.pow(w, 2)) |> Nx.to_number()
      assert 10 * :math.log10(signal / mse) > 3.0
    end

    test "default is twn mode", %{w: w} do
      {_, _, default_delta} = Quantizer.quantize(w)
      {_, _, twn_delta} = Quantizer.quantize(w, mode: :twn)
      assert default_delta == twn_delta
    end
  end
end
