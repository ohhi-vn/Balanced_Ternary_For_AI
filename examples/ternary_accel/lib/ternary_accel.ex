defmodule TernaryAccel do
  @moduledoc """
  Balanced-ternary inference on GPU via Nx — NVIDIA CUDA (EXLA) and
  Apple Silicon MLX (EMLX).

  The demo runs the full ternary pipeline on the selected device:

    1. Ternarize weights on-device with the BitNet b1.58 **absmean** rule
       (see `concept.md` → "Quantization Theory")
    2. Store the trits as packed u8 codes `{0, 1, 2}` — 1 byte/weight
       instead of 4 bytes FP32
    3. Decode trits *on device* inside a compiled kernel and run GEMM:
       this is the software analogue of the hardware add/sub/skip decoder
    4. Verify accuracy against the FP32 reference
    5. Benchmark dense vs ternary GEMM end-to-end

  On current GPUs the win is storage + bandwidth (4× with byte codes,
  ~6.3× with the tool's 10-trits-per-16-bit packing); the multiplier-free
  compute path requires the custom accelerator described in
  `details/12-custom-ternary-accelerator-design.md`.

  ## Usage

      mix run -e "TernaryAccel.run()"                     # auto-detect backend
      mix run -e "TernaryAccel.run(backend: :cuda)"       # force CUDA (EXLA)
      mix run -e "TernaryAccel.run(backend: :mlx)"        # force Apple MLX
      mix run -e "TernaryAccel.run(backend: :exla_cpu)"   # XLA on CPU
      mix run -e "TernaryAccel.run(backend: :binary)"     # plain CPU

  ## Backend requirements

    - `:cuda` — `{:exla, "~> 0.12"}` compiled with CUDA support:
      `XLA_TARGET=cuda120 mix deps.get && mix compile`
    - `:mlx` — `{:emlx, "~> 0.4"}`, macOS on Apple Silicon only
  """

  import Nx.Defn

  @default_opts [
    rows: 2048,
    cols: 4096,
    batch: 256,
    iters: 20
  ]

  # The pure-Elixir fallback has no compiled kernels; shrink the demo
  # matrices unless the user asked for explicit sizes.
  @binary_opts [
    rows: 512,
    cols: 1024,
    batch: 64,
    iters: 5
  ]

  # ── Entry point ──────────────────────────────────────────────

  def run(opts \\ []) do
    backend = select_backend(Keyword.get(opts, :backend, :auto))

    defaults =
      if backend.name == :binary and Keyword.take(opts, [:rows, :cols, :batch, :iters]) == [] do
        @binary_opts
      else
        @default_opts
      end

    opts = Keyword.merge(defaults, opts)

    print_header(backend, opts)

    Nx.default_backend(backend.nx_backend)

    case backend.compiler do
      nil -> :ok
      compiler -> Nx.Defn.default_options(compiler: compiler)
    end

    rows = opts[:rows]
    cols = opts[:cols]

    key = Nx.Random.key(42)
    {w, _} = Nx.Random.normal(key, 0.0, 0.02, shape: {rows, cols}, type: :f32)

    # Steps 1+2: quantize on device → packed u8 codes + shared scale
    {codes, scales, gamma} = quantize_absmean(w)

    IO.puts("Absmean γ (shared scale):        #{Float.round(Nx.to_number(gamma), 6)}")
    IO.puts("Packed weight sparsity:          #{Float.round(zero_fraction(codes) * 100, 2)}% zeros")
    IO.puts("Weight memory: #{rows * cols * 4} B (f32) → #{rows * cols} B (u8 codes), 4× smaller")

    # Step 3: forward passes through both representations
    {x, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {opts[:batch], cols}, type: :f32)

    y_ref = dense_forward(x, w)
    y_tri = ternary_forward(x, codes, scales)

    # Step 4: accuracy vs FP32 reference
    cos = cosine(y_ref, y_tri)
    IO.puts("\n=== Accuracy vs FP32 reference ===")
    IO.puts("Cosine similarity of outputs:    #{Float.round(cos, 5)}")

    # Step 5: throughput + traffic report
    benchmark(x, w, codes, scales, opts, backend)

    :ok
  end

  # ── Ternary kernels (compiled per-backend by Nx.Defn) ────────

  @doc """
  BitNet b1.58 absmean quantization:

      γ = mean(|W|)
      T = roundclip(W / γ, -1, +1)

  Returns `{codes, scales, gamma}` where codes are u8 in `{0,1,2}`
  representing trits `{-1, 0, +1}` — the storage format — and every
  output channel shares the single scale γ.
  """
  defn quantize_absmean(w) do
    gamma = Nx.mean(Nx.abs(w))

    # Add 1 BEFORE converting to u8: some backends wrap (not clamp)
    # negative floats on integer conversion, so {-1,0,+1} must become
    # {0,1,2} while still in floating point.
    t = w |> Nx.divide(gamma) |> Nx.round() |> Nx.clip(-1.0, 1.0)
    codes = t |> Nx.add(1.0) |> Nx.as_type(:u8)
    scales = Nx.broadcast(gamma, {elem(Nx.shape(w), 0)})
    {codes, scales, gamma}
  end

  @doc """
  Decode u8 codes back to scaled ternary weights, on device.

  This is the software stand-in for the hardware trit decoder: the
  multiply-by-scale is folded into the decode step so the GEMM itself
  sees {-α, 0, +α} values exactly like the ASIC datapath.
  """
  defn decode_weights(codes, scales) do
    codes
    |> Nx.as_type(:f32)
    |> Nx.subtract(1.0)
    |> Nx.multiply(Nx.new_axis(scales, 1))
  end

  @doc """
  Dense FP32 forward: y = x · Wᵀ.
  """
  defn dense_forward(x, w) do
    Nx.dot(x, Nx.transpose(w))
  end

  @doc """
  Ternary forward from packed codes: decode then y = x · Wᵀ.

  Note the dot product is unchanged — on current GPUs we recover the
  bandwidth advantage; on the target ASIC the same codes drive an
  add/sub/skip array with no multiplier at all.
  """
  defn ternary_forward(x, codes, scales) do
    w = decode_weights(codes, scales)
    Nx.dot(x, Nx.transpose(w))
  end

  @doc """
  Reduction forms used for fair timing: forces full computation and
  returns a scalar so each timed iteration synchronizes the device.
  """
  defn bench_dense_step(x, w) do
    x |> Nx.dot(Nx.transpose(w)) |> Nx.sum()
  end

  defn bench_ternary_step(x, codes, scales) do
    w = decode_weights(codes, scales)
    x |> Nx.dot(Nx.transpose(w)) |> Nx.sum()
  end

  # ── Benchmark ────────────────────────────────────────────────

  defp benchmark(x, w, codes, scales, opts, backend) do
    iters = opts[:iters]
    flops = 2.0 * elem(Nx.shape(x), 0) * elem(Nx.shape(w), 0) * elem(Nx.shape(w), 1)

    IO.puts("\n=== Benchmark (#{iters} iterations, incl. device sync) ===")

    traffic_dense = elem(Nx.shape(w), 0) * elem(Nx.shape(w), 1) * 4
    traffic_tern = elem(Nx.shape(w), 0) * elem(Nx.shape(w), 1)

    if backend.name == :binary do
      IO.puts(
        "Skipped: the pure-Elixir CPU fallback has no compiled kernels, so timings\n" <>
          "would only measure interpreter overhead. Run with a GPU backend for real numbers."
      )
    else
      # Warmup first: the initial call also JIT-compiles the kernels for
      # the active backend, which would otherwise dominate iteration 1.
      bench_dense_step(x, w) |> sync_to_host()
      bench_ternary_step(x, codes, scales) |> sync_to_host()

      # Each timed call ends in a scalar readback, forcing full
      # computation before the clock stops — timings are honest
      # end-to-end latencies including device synchronization, not
      # kernel-launch-only figures.
      t_dense = loop_time(iters, fn -> bench_dense_step(x, w) end)
      t_tern = loop_time(iters, fn -> bench_ternary_step(x, codes, scales) end)

      tflops = fn us -> flops / (us * 1.0e6) end

      IO.puts(
        "Dense  FP32 GEMM: #{fmt_us(t_dense)} us/iter  (~#{Float.round(tflops.(t_dense), 2)} TFLOP/s)"
      )

      IO.puts(
        "Ternary u8  GEMM: #{fmt_us(t_tern)} us/iter  (~#{Float.round(tflops.(t_tern), 2)} TFLOP/s)"
      )

      speedup = if t_tern > 0, do: t_dense / t_tern, else: 0.0
      IO.puts("End-to-end ratio:                #{Float.round(speedup, 2)}×")
    end

    IO.puts("\nPer-GEMM weight traffic:")
    IO.puts("  FP32:            #{format_bytes(traffic_dense)}")
    IO.puts("  u8 codes:        #{format_bytes(traffic_tern)} (#{div(traffic_dense, traffic_tern)}× less)")
    IO.puts("  10→16 packing:   #{format_bytes(div(traffic_tern, 10) * 2)} theoretical (.tbin format)")
  end

  defp loop_time(iters, fun) do
    times =
      for _ <- 1..iters do
        {us, scalar} = :timer.tc(fun)
        sync_to_host(scalar)
        us
      end

    Enum.sum(times) / length(times)
  end

  # Forces pending device work to complete by materializing the scalar
  # on the host — the synchronization point for benchmark timing.
  defp sync_to_host(scalar_or_tensor), do: Nx.to_number(scalar_or_tensor)

  # ── Reporting helpers ────────────────────────────────────────

  # Fraction of zero trits. In code space {0,1,2} the zero trit is 1.
  defp zero_fraction(codes) do
    codes
    |> Nx.equal(1)
    |> Nx.mean()
    |> Nx.to_number()
  end

  defp cosine(a, b) do
    num =
      a
      |> Nx.multiply(b)
      |> Nx.sum()
      |> Nx.to_number()

    na = :math.sqrt(a |> Nx.pow(2) |> Nx.sum() |> Nx.to_number())
    nb = :math.sqrt(b |> Nx.pow(2) |> Nx.sum() |> Nx.to_number())
    num / max(na * nb, 1.0e-9)
  end

  defp fmt_us(us) when us < 1000, do: :erlang.float_to_binary(us, decimals: 1)
  defp fmt_us(us), do: :erlang.float_to_binary(us, decimals: 0)

  defp format_bytes(bytes) when bytes >= 1024 * 1024,
    do: "#{Float.round(bytes / 1024 / 1024, 2)} MB"

  defp format_bytes(bytes) when bytes >= 1024, do: "#{Float.round(bytes / 1024, 1)} KB"
  defp format_bytes(bytes), do: "#{bytes} B"

  # ── Backend selection ────────────────────────────────────────

  defp print_header(backend, opts) do
    IO.puts("=== TernaryAccel — balanced ternary on Nx ===\n")
    IO.puts("Backend: #{backend.name}  (#{backend.platform})")

    unless backend.available do
      IO.puts("""
      WARNING: requested backend not usable here — falling back to Nx.BinaryBackend.
      For CUDA: {:exla, "~> 0.12"} + XLA_TARGET=cuda120 mix deps.get compile
      For MLX:  {:emlx, "~> 0.4"} on Apple Silicon macOS
      """)
    end

    IO.puts(
      "GEMM shape: {#{opts[:batch]}, #{opts[:cols]}} × {#{opts[:cols]}, #{opts[:rows]}}, #{opts[:iters]} iterations\n"
    )
  end

  @doc """
  Resolve the effective backend configuration.

  Order for `:auto`: CUDA → MLX → EXLA/CPU → BinaryBackend — fastest
  device first, with the always-available CPU fallback last. Any other
  value probes just that backend and falls back to binary when
  unavailable.
  """
  def select_backend(choice) do
    candidates =
      case choice do
        :auto -> [:cuda, :mlx, :exla_cpu, :binary]
        other -> [other]
      end

    candidates
    |> Enum.map(&probe/1)
    |> Enum.find(& &1.available)
    |> Kernel.||(probe(:binary))
  rescue
    _ -> probe(:binary)
  catch
    _, _ -> probe(:binary)
  end

  defp probe(:binary) do
    %{
      name: :binary,
      platform: "CPU (Nx.BinaryBackend)",
      nx_backend: Nx.BinaryBackend,
      # Eager evaluation — no XLA compile step, so the fallback demo
      # starts instantly even on machines without GPU toolchains.
      compiler: Nx.Defn.Evaluator,
      available: true
    }
  end

  defp probe(:exla_cpu) do
    if Code.ensure_loaded?(EXLA.Backend) and probe_transfer({EXLA.Backend, client: :host}) do
      %{
        name: :exla_cpu,
        platform: "CPU (XLA)",
        nx_backend: {EXLA.Backend, client: :host},
        compiler: EXLA.Compiler,
        available: true
      }
    else
      %{probe(:binary) | name: :exla_cpu, available: false}
    end
  end

  defp probe(:cuda) do
    if Code.ensure_loaded?(EXLA.Backend) and cuda_supported?() and
         probe_transfer({EXLA.Backend, client: :cuda}) do
      %{
        name: :cuda,
        platform: "NVIDIA CUDA (XLA)",
        nx_backend: {EXLA.Backend, client: :cuda},
        compiler: EXLA.Compiler,
        available: true
      }
    else
      %{probe(:binary) | name: :cuda, available: false}
    end
  end

  defp probe(:mlx) do
    if Code.ensure_loaded?(EMLX.Backend) and probe_transfer(EMLX.Backend) do
      %{
        name: :mlx,
        platform: "Apple Silicon Metal (MLX/EMLX)",
        nx_backend: EMLX.Backend,
        compiler: EMLX,
        available: true
      }
    else
      %{probe(:binary) | name: :mlx, available: false}
    end
  end

  defp probe(other), do: %{probe(:binary) | name: other, available: false}

  # Ask XLA which platforms have devices *before* creating a client —
  # attempting a CUDA client on a machine without NVIDIA drivers kills
  # the EXLA.Client process instead of raising.
  defp cuda_supported? do
    platforms = EXLA.Client.get_supported_platforms()

    match?(%{cuda: n} when is_integer(n) and n > 0, platforms)
  rescue
    _ -> false
  catch
    _, _ -> false
  end

  defp probe_transfer(backend) do
    [1.0]
    |> Nx.tensor(backend: backend)
    |> Nx.sum()
    |> Nx.to_number()

    true
  rescue
    _ -> false
  catch
    _, _ -> false
  end
end
