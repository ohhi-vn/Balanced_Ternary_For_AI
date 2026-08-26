defmodule TernaryAccel.MixProject do
  use Mix.Project

  def project do
    [
      app: :ternary_accel,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  # Both GPU backends are optional so the example compiles anywhere;
  # TernaryAccel detects at runtime what is actually usable:
  #
  #   - EXLA  → NVIDIA CUDA GPUs (set XLA_TARGET=cuda120 before fetching)
  #   - EMLX  → Apple Silicon (Metal via Apple MLX); macOS + arm64 only
  #
  # Without either, everything falls back to Nx.BinaryBackend (CPU).
  defp deps do
    [
      {:nx, "~> 0.12"},
      {:exla, "~> 0.12", optional: true},
      {:emlx, "~> 0.4", optional: true}
    ]
  end
end
