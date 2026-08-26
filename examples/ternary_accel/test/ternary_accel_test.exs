defmodule TernaryAccelTest do
  use ExUnit.Case, async: true

  # Exercise the ternary kernels on the plain CPU binary backend so the
  # suite runs anywhere. GPU backends (CUDA/MLX) are exercised by
  # running the demo manually on capable hardware.

  test "absmean quantization matches the BitNet b1.58 rule" do
    Nx.default_backend(Nx.BinaryBackend)

    w = Nx.tensor([[0.4, -0.2, 0.05, 0.3, -0.5, 0.0]])
    {codes, scales, gamma} = TernaryAccel.quantize_absmean(w)

    expected_gamma = w |> Nx.abs() |> Nx.mean() |> Nx.to_number()
    assert_in_delta Nx.to_number(gamma), expected_gamma, 1.0e-6

    assert Nx.to_flat_list(scales) == List.duplicate(expected_gamma, 1)

    # W/γ = [1.66, -0.83, 0.21, 1.24, -2.07, 0] → round+clip [1, -1, 0, 1, -1, 0]
    assert Nx.to_flat_list(codes) == [2, 0, 1, 2, 0, 1]
  end

  test "decode_weights restores {-α, 0, +α}" do
    codes = Nx.tensor([[2, 1, 0]])
    scales = Nx.tensor([0.5])

    decoded = TernaryAccel.decode_weights(codes, scales)
    assert Nx.to_flat_list(decoded) == [0.5, 0.0, -0.5]
  end

  test "ternary forward approximates dense forward" do
    Nx.default_backend(Nx.BinaryBackend)

    key = Nx.Random.key(42)
    {w, _} = Nx.Random.normal(key, 0.0, 0.02, shape: {32, 64})
    {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {4, 64})

    {codes, scales, _gamma} = TernaryAccel.quantize_absmean(w)

    y_ref = TernaryAccel.dense_forward(x, w)
    y_tri = TernaryAccel.ternary_forward(x, codes, scales)

    cos = cosine(y_ref, y_tri)
    # Tiny weights with absmean quantization keep outputs highly aligned
    assert cos > 0.85
  end

  test "run/1 completes end-to-end on the binary backend" do
    assert :ok =
             TernaryAccel.run(
               backend: :binary,
               rows: 64,
               cols: 128,
               batch: 8,
               iters: 2
             )
  end

  defp cosine(a, b) do
    num = a |> Nx.multiply(b) |> Nx.sum() |> Nx.to_number()

    na = :math.sqrt(a |> Nx.pow(2) |> Nx.sum() |> Nx.to_number())
    nb = :math.sqrt(b |> Nx.pow(2) |> Nx.sum() |> Nx.to_number())
    num / max(na * nb, 1.0e-9)
  end
end
