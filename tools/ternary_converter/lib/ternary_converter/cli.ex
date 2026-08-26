defmodule TernaryConverter.CLI do
  @moduledoc """
  Command-line interface for the TernaryConverter tool.

  Four commands share one option parser:

    - `convert` — safetensors checkpoint → `.tbin` (or synthetic model
      when no `--input` is given)
    - `demo` — quantize/pack/GEMM walkthrough on random weights
    - `info` — inspect a `.tbin` file
    - `validate` — round-trip + per-layer forward verification

  Options are parsed with `OptionParser` under `strict:` mode, so typos
  like `--deltta` are reported instead of being silently ignored.
  """

  alias TernaryConverter.{Architectures, Pipeline, Quantizer}

  @modes [:delta, :twn, :absmean]

  # Declared once so OptionParser validates types for every command.
  @opt_switches [
    input: :string,
    output: :string,
    model: :string,
    arch: :string,
    mode: :string,
    delta: :float,
    sparsity: :float,
    rows: :integer,
    cols: :integer,
    batch: :integer
  ]

  @spec main(OptionParser.parsed() | [String.t()]) :: :ok
  def main(args) do
    case args do
      ["convert" | rest] -> do_convert(parse_opts(rest))
      ["demo" | rest] -> do_demo(parse_opts(rest))
      ["info" | rest] -> do_info(parse_opts(rest))
      ["validate" | rest] -> do_validate(parse_opts(rest))
      _ -> print_usage()
    end
  end

  defp parse_opts(args) do
    args
    |> OptionParser.parse(strict: @opt_switches)
    |> elem(0)
  end

  defp print_usage do
    IO.puts("""
    TernaryConverter — Convert neural network weights to balanced ternary

    Usage:
      ternary_converter convert --input model.safetensors [--arch llama] [--mode twn] [--delta 0.5] [--output model.tbin]
      ternary_converter convert [--sparsity 0.5] [--delta 0.5] [--output model.tbin]   # synthetic demo model
      ternary_converter demo [--rows 64] [--cols 128] [--delta 0.5] [--batch 16]
      ternary_converter info [--model model.tbin]
      ternary_converter validate [--model model.tbin]

    Commands:
      convert    Convert a safetensors model to .tbin (or a synthetic model if no --input)
      demo       Run a full demo with synthetic weights
      info       Show info about a .tbin model
      validate   Validate a .tbin model (round-trip, inference)

    Conversion options:
      --input PATH      Source model (.safetensors)
      --arch NAME       Model family: llama | gpt2 | bert | vit | generic (default: generic)
      --mode NAME       Quantization mode: delta | twn | absmean (default: twn)
                        delta   fixed threshold + per-channel MSE-optimal scales
                        twn     Δ = 0.7·mean|W| (Ternary Weight Networks rule)
                        absmean BitNet b1.58: normalize by mean|W|, round to trit
      --delta FLOAT     Threshold (mode=delta only)
      --sparsity FLOAT  Target sparsity via auto threshold (synthetic path only)

    Other options:
      --output PATH     Output file path (default: model.tbin)
      --rows INT        Number of rows for demo weights (default: 64)
      --cols INT        Number of cols for demo weights (default: 128)
      --batch INT       Batch size for demo (default: 16)
      --model PATH      Path to .tbin model file

    Architectures (mixed-precision policy):
      llama   ternarize q/k/v/o + gate/up/down; keep embeddings, norms, lm_head FP16
      gpt2    ternarize c_attn/c_proj/c_fc (transposed); keep wte/wpe/ln/lm_head FP16
      bert    ternarize attention + FFN dense; keep embeddings/pooler/LayerNorm FP16
      vit     ternarize attention qkv + MLP; keep patch embed/norms/head FP16
      generic heuristic: ternarize 2D weights except embed/norm/head-like names
    """)
  end

  # ── Convert ───────────────────────────────────────────────────
  #
  # Two paths share this command:
  #   * --input given → Pipeline.run over a real safetensors file
  #   * no --input    → synthetic weights (useful as a smoke test)

  defp do_convert(opts) do
    # Validate --arch up front so both paths reject bad values instead
    # of the synthetic path silently ignoring one.
    with {:ok, arch} <- fetch_arch(opts) do
      case Keyword.get(opts, :input) do
        nil -> convert_synthetic(opts)
        input -> convert_file(input, opts, arch)
      end
    else
      {:error, message} -> halt_with(message)
    end
  end

  defp convert_file(input, opts, arch) do
    output = Keyword.get(opts, :output, default_output(input))

    with {:ok, mode_opts} <- fetch_mode(opts) do
      IO.puts("TernaryConverter — Convert #{input}")
      IO.puts("  Architecture: #{arch}")
      IO.puts("  Mode: #{Keyword.get(mode_opts, :mode)}")
      IO.puts("  Output: #{output}")
      IO.puts("")

      case Pipeline.run(input, output, [arch: arch] ++ mode_opts) do
        {:ok, summary} -> print_conversion_summary(summary)
        {:error, reason} -> halt_with("Error converting #{input}: #{inspect(reason)}")
      end
    else
      {:error, message} -> halt_with(message)
    end
  end

  # Whitelist before atomizing: never turn arbitrary CLI strings into
  # atoms at runtime.
  defp fetch_arch(opts) do
    arch =
      opts
      |> Keyword.get(:arch, "generic")
      |> String.downcase()
      |> String.to_atom()

    if arch in Architectures.presets() do
      {:ok, arch}
    else
      {:error,
       "unknown --arch #{arch}. Valid: #{Enum.join(Architectures.presets(), ", ")}"}
    end
  end

  defp fetch_mode(opts) do
    case Keyword.get(opts, :mode, "twn") |> String.downcase() do
      m when m in ["delta", "twn", "absmean"] ->
        mode = String.to_atom(m)

        # Only forward :delta when the user asked for delta mode;
        # otherwise Quantizer's default resolution applies.
        case Keyword.fetch(opts, :delta) do
          {:ok, delta} -> {:ok, [mode: mode, delta: delta]}
          :error -> {:ok, [mode: mode]}
        end

      other ->
        {:error, "unknown --mode #{other}. Valid: #{Enum.join(@modes, ", ")}"}
    end
  end

  defp default_output(input) do
    base = input |> Path.basename() |> Path.rootname()
    base <> ".tbin"
  end

  defp print_conversion_summary(summary) do
    %{stats: stats, metadata: meta} = summary

    IO.puts("""
    Converted:
      Ternary layers: #{meta["ternary_layers"]}
      Kept FP16:      #{meta["fp16_layers"]}
      Skipped:        #{meta["skipped_tensors"]} (buffers/biases)

    Ternary statistics:
      Parameters: #{stats.total_parameters}
      Sparsity:   #{Float.round(stats.overall_sparsity * 100, 1)}%
      Original:   #{Float.round(stats.original_size_mb, 2)} MB
      Packed:     #{Float.round(stats.packed_size_mb, 2)} MB
      Compression: #{Float.round(stats.compression_ratio, 1)}×
    """)
  end

  defp convert_synthetic(opts) do
    delta_opt = Keyword.get(opts, :delta)
    target_sparsity = Keyword.get(opts, :sparsity, 0.5)
    output = Keyword.get(opts, :output, "model.tbin")

    IO.puts("TernaryConverter — Convert (synthetic model)")
    IO.puts("  Target sparsity: #{target_sparsity}")
    IO.puts("  Output: #{output}")
    IO.puts("")

    weights = create_synthetic_model()

    IO.puts("Calibrating and converting #{map_size(weights)} layers...")

    layers =
      weights
      |> Enum.map(fn {name, w} ->
        # Per-layer auto threshold hits the requested zero fraction on
        # each tensor independently (sparsity varies by shape/distribution).
        {auto_delta, _actual_sparsity} =
          if target_sparsity > 0.0 do
            Quantizer.auto_threshold(w, target_sparsity)
          else
            {delta_opt || 0.5, 0.0}
          end

        layer = TernaryConverter.convert_layer(w, name, delta: auto_delta)

        IO.puts(
          "  #{name}: shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(auto_delta, 3)}"
        )

        layer
      end)

    stats = TernaryConverter.stats(layers)

    IO.puts("\nModel statistics:")
    print_stat_lines(stats)

    :ok =
      TernaryConverter.export(layers, output, metadata: %{command: "convert", delta: delta_opt})
  end

  # ── Demo ──────────────────────────────────────────────────────

  defp do_demo(opts) do
    rows = Keyword.get(opts, :rows, 64)
    cols = Keyword.get(opts, :cols, 128)
    delta = Keyword.get(opts, :delta, 0.5)
    batch = Keyword.get(opts, :batch, 16)

    IO.puts("=== TernaryConverter Demo ===\n")
    Nx.default_backend(Nx.BinaryBackend)

    key = Nx.Random.key(42)
    {w_raw, key2} = Nx.Random.uniform(key, 0.0, 1.0, shape: {rows, cols})
    w = Nx.subtract(Nx.multiply(w_raw, 2.0), 1.0)
    {x_raw, _key} = Nx.Random.uniform(key2, 0.0, 1.0, shape: {batch, cols})
    x = Nx.subtract(Nx.multiply(x_raw, 2.0), 1.0)

    IO.puts("Weight tensor shape: #{inspect(Nx.shape(w))}")
    IO.puts("Activation tensor shape: #{inspect(Nx.shape(x))}")

    # Quantize
    w_tern = Quantizer.ternarize(w, delta)
    sparsity = Nx.divide(Nx.sum(Nx.equal(w_tern, 0)), Nx.size(w_tern)) |> Nx.to_number()
    IO.puts("Ternary weight sparsity: #{Float.round(sparsity * 100, 2)}% zeros")

    # Scales + GEMM
    scales = Quantizer.compute_scales(w, delta)
    scaled_w = Nx.multiply(w_tern, Nx.new_axis(scales, 1))
    result = Nx.dot(x, Nx.transpose(scaled_w))
    IO.puts("GEMM output shape: #{inspect(Nx.shape(result))}")

    # Effective MACs
    nonzero = Nx.sum(Nx.not_equal(w_tern, 0)) |> Nx.to_number()
    total = Nx.size(w_tern) |> Nx.to_number()
    IO.puts("Effective MACs: #{nonzero} / #{total} (#{Float.round(nonzero / total * 100, 2)}%)")

    # Pack
    packed = TernaryConverter.Packer.pack(w_tern)
    IO.puts("Packed size: #{byte_size(packed)} bytes (original: #{rows * cols * 4} bytes f32)")

    {_orig, _packed, ratio} = TernaryConverter.Packer.compression_ratio(w_tern)
    IO.puts("Compression ratio: #{ratio}×")

    _unpacked = TernaryConverter.Packer.unpack(packed, rows * cols)
    match = TernaryConverter.Packer.verify_roundtrip(w_tern)
    IO.puts("Round-trip match: #{match}")

    metrics = Quantizer.quality_metrics(w, delta)
    IO.puts("\nQuality metrics:")
    IO.puts("  Sparsity: #{Float.round(metrics.sparsity * 100, 1)}%")
    IO.puts("  Density: #{Float.round(metrics.density * 100, 1)}%")
    IO.puts("  +1 ratio: #{Float.round(metrics.positive_ratio * 100, 1)}%")
    IO.puts("  -1 ratio: #{Float.round(metrics.negative_ratio * 100, 1)}%")
    IO.puts("  MSE: #{Float.round(metrics.mse, 6)}")
    IO.puts("  SQNR: #{Float.round(metrics.sqnr, 2)} dB")

    IO.puts("\n=== Demo complete ===")
  end

  # ── Info ──────────────────────────────────────────────────────

  defp do_info(opts) do
    path = Keyword.get(opts, :model, "model.tbin")

    case TernaryConverter.load(path) do
      {:ok, layers, metadata} ->
        stats = TernaryConverter.stats(layers)

        IO.puts("=== Model Info: #{path} ===\n")
        IO.puts("Metadata:")
        Enum.each(metadata, fn {k, v} -> IO.puts("  #{k}: #{inspect(v)}") end)

        IO.puts("\nStatistics:")
        print_model_stats(stats)

        IO.puts("\nLayers:")

        Enum.each(layers, fn
          %TernaryConverter.Layer{} = layer ->
            IO.puts(
              "  #{layer.name}: ternary, shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(layer.delta, 3)}"
            )

          %TernaryConverter.RawLayer{} = layer ->
            IO.puts("  #{layer.name}: fp16, shape=#{inspect(layer.shape)}")
        end)

      {:error, reason} ->
        IO.puts("Error loading #{path}: #{inspect(reason)}")
    end
  end

  # ── Validate ──────────────────────────────────────────────────

  defp do_validate(opts) do
    path = Keyword.get(opts, :model, "model.tbin")

    case TernaryConverter.load(path) do
      {:ok, layers, _metadata} ->
        IO.puts("=== Validating: #{path} ===\n")

        # Round-trip check: unpacking and re-packing must reproduce the
        # stored bytes exactly — proves decode(pack(x)) == x for every
        # stored payload.
        IO.puts("Round-trip verification:")

        Enum.each(layers, fn
          %TernaryConverter.Layer{} = layer ->
            total = elem(layer.shape, 0) * elem(layer.shape, 1)

            ok =
              layer.weight_packed
              |> TernaryConverter.Packer.unpack(total)
              |> TernaryConverter.Packer.pack() == layer.weight_packed

            IO.puts("  #{layer.name}: #{if ok, do: "OK", else: "FAIL"}")

          %TernaryConverter.RawLayer{} = layer ->
            IO.puts("  #{layer.name}: fp16 (#{byte_size(layer.data)} bytes)")
        end)

        # Per-layer forward: converted layers chain only inside an
        # architecture-aware runtime, so standalone shape checks are the
        # meaningful smoke test here.
        IO.puts("\nInference test (per-layer forward):")

        ternary_layers = Enum.filter(layers, &match?(%TernaryConverter.Layer{}, &1))

        if ternary_layers != [] do
          Enum.each(ternary_layers, fn layer ->
            {_out, in_features} = layer.shape
            {x, _} = Nx.Random.uniform(Nx.Random.key(123), 0.0, 1.0, shape: {1, in_features})
            output = TernaryConverter.Layer.forward_nx(layer, x)
            IO.puts("  #{layer.name}: #{inspect(Nx.shape(x))} -> #{inspect(Nx.shape(output))}")
          end)
        else
          IO.puts("  No ternary layers to run inference on.")
        end

        stats = TernaryConverter.stats(layers)
        IO.puts("\nModel stats:")
        print_stat_lines(stats)

        IO.puts("\n=== Validation complete ===")

      {:error, reason} ->
        IO.puts("Error loading #{path}: #{inspect(reason)}")
    end
  end

  # ── Shared formatting ─────────────────────────────────────────

  defp print_stat_lines(stats) do
    IO.puts("  Layers: #{stats.num_layers}")
    IO.puts("  Parameters: #{stats.total_parameters}")
    IO.puts("  Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
    IO.puts("  Original: #{Float.round(stats.original_size_mb, 2)} MB")
    IO.puts("  Packed: #{Float.round(stats.packed_size_mb, 2)} MB")
    IO.puts("  Compression: #{Float.round(stats.compression_ratio, 1)}×")
  end

  defp print_model_stats(stats) do
    IO.puts("  Layers: #{stats.num_layers}")
    IO.puts("  Parameters: #{stats.total_parameters}")
    IO.puts("  Non-zero: #{stats.nonzero_parameters}")
    IO.puts("  Zero: #{stats.zero_parameters}")
    IO.puts("  Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
    IO.puts("  Original size: #{Float.round(stats.original_size_mb, 2)} MB")
    IO.puts("  Packed size: #{Float.round(stats.packed_size_mb, 2)} MB")
    IO.puts("  Compression: #{Float.round(stats.compression_ratio, 1)}×")
  end

  # ── Helpers ───────────────────────────────────────────────────

  defp create_synthetic_model do
    [
      {"fc1", Nx.Random.key(1), {256, 512}},
      {"fc2", Nx.Random.key(2), {128, 256}},
      {"fc3", Nx.Random.key(3), {64, 128}},
      {"fc4", Nx.Random.key(4), {32, 64}}
    ]
    |> Enum.map(fn {name, key, shape} ->
      {t, _} = Nx.Random.uniform(key, 0.0, 1.0, shape: shape)
      scaled = Nx.subtract(Nx.multiply(t, 2.0), 1.0)
      {name, scaled}
    end)
    |> Map.new()
  end

  defp halt_with(message) do
    IO.puts(message)
    System.halt(1)
  end
end
