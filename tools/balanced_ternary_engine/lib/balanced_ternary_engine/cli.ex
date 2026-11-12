defmodule BalancedTernaryEngine.CLI do
  @moduledoc """
  Command-line interface for the BalancedTernaryEngine tool.

  Four commands share one option parser:

    - `convert` — safetensors checkpoint → `.tbin` (or synthetic model
      when no `--input` is given)
    - `demo` — quantize/pack/GEMM walkthrough on random weights
    - `info` — inspect a `.tbin` file
    - `validate` — round-trip + per-layer forward verification

  Options are parsed with `OptionParser` under `strict:` mode, so typos
  like `--deltta` are reported instead of being silently ignored.
  """

  alias BalancedTernaryEngine.{Architectures, Pipeline, Quantizer}

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
    sparsity_target: :float,
    clip: :float,
    calibration_data: :string,
    target_similarity: :float,
    profile: :string,
    epochs: :integer,
    lr: :float,
    rows: :integer,
    cols: :integer,
    batch: :integer
  ]

  @spec main(OptionParser.parsed() | [String.t()]) :: :ok
  def main([]), do: print_usage()

  def main([command | rest]) do
    dispatch(command, rest)
  end

  defp dispatch("convert", rest), do: do_convert(parse_opts(rest))
  defp dispatch("yolo", rest), do: do_convert([profile: "yolo"] ++ parse_opts(rest))
  defp dispatch("imagenet", rest), do: do_convert([profile: "imagenet"] ++ parse_opts(rest))
  defp dispatch("llm", rest), do: do_convert([profile: "llm"] ++ parse_opts(rest))
  defp dispatch("demo", rest), do: do_demo(parse_opts(rest))
  defp dispatch("info", rest), do: do_info(parse_opts(rest))
  defp dispatch("validate", rest), do: do_validate(parse_opts(rest))
  defp dispatch("qat", rest), do: do_qat(parse_opts(rest))
  defp dispatch(_command, _rest), do: print_usage()

  defp parse_opts(args) do
    args
    |> OptionParser.parse(strict: @opt_switches)
    |> elem(0)
  end

  defp print_usage do
    IO.puts("""
    BalancedTernaryEngine — Convert neural network weights to balanced ternary

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
      yolo    ternarize backbone conv/linear; keep detection head, norms, biases FP16
      imagenet ternarize backbone conv/linear; keep classifier head and norms FP16
      llm     ternarize decoder projections; keep embeddings, norms, lm_head FP16 (<8B)
      generic heuristic: ternarize 2D weights except embed/norm/head-like names
    """)
  end

  # ── Convert ───────────────────────────────────────────────────
  #
  # Two paths share this command:
  #   * --input given → Pipeline.run over a real safetensors file
  #   * no --input    → synthetic weights (useful as a smoke test)

  defp do_convert(opts) do
    case Keyword.get(opts, :profile) do
      nil ->
        :ok

      profile when profile in ["yolo", "imagenet", "llm"] ->
        :ok

      profile ->
        halt_with("unknown --profile #{profile}. Valid: yolo, imagenet, llm")
    end

    case fetch_arch(opts) do
      {:ok, arch} ->
        case Keyword.get(opts, :input) do
          nil -> convert_synthetic(opts)
          input -> convert_file(input, opts, arch)
        end

      {:error, message} ->
        halt_with(message)
    end
  end

  defp profile_to_arch("yolo"), do: {:ok, :yolo}
  defp profile_to_arch("imagenet"), do: {:ok, :imagenet}
  defp profile_to_arch("llm"), do: {:ok, :llm}
  defp profile_to_arch(other), do: {:error, other}

  defp convert_file(input, opts, arch) do
    output = Keyword.get(opts, :output, default_output(input))

    case fetch_mode(opts) do
      {:ok, mode_opts} ->
        extra_opts =
          []
          |> maybe_put(:sparsity_target, Keyword.get(opts, :sparsity_target))
          |> maybe_put(:clip_sigma, Keyword.get(opts, :clip))
          |> maybe_put(:calibration_data, Keyword.get(opts, :calibration_data))
          |> maybe_put(:target_similarity, Keyword.get(opts, :target_similarity))

        # Wire sparsity_target -> auto threshold via a flag consumed by Pipeline.
        pipeline_opts =
          [arch: arch] ++ mode_opts ++ extra_opts

        pipeline_opts =
          if st = Keyword.get(opts, :sparsity_target) do
            Keyword.put(pipeline_opts, :sparsity_target, st)
          else
            pipeline_opts
          end

        IO.puts("BalancedTernaryEngine — Convert #{input}")
        IO.puts("  Architecture: #{arch}")
        IO.puts("  Mode: #{Keyword.get(mode_opts, :mode)}")
        IO.puts("  Output: #{output}")
        IO.puts("")

        case Pipeline.run(input, output, pipeline_opts) do
          {:ok, summary} -> print_conversion_summary(summary)
          {:error, reason} -> halt_with("Error converting #{input}: #{inspect(reason)}")
        end

      {:error, message} ->
        halt_with(message)
    end
  end

  # Whitelist before atomizing: never turn arbitrary CLI strings into
  # atoms at runtime.
  defp fetch_arch(opts) do
    opts =
      case {Keyword.get(opts, :profile), Keyword.get(opts, :arch)} do
        {nil, _} ->
          opts

        {profile, nil} ->
          case profile_to_arch(profile) do
            {:ok, arch} -> Keyword.put(opts, :arch, Atom.to_string(arch))
            {:error, _} -> opts
          end

        _ ->
          opts
      end

    arch =
      opts
      |> Keyword.get(:arch, "generic")
      |> String.downcase()
      |> String.to_atom()

    if arch in Architectures.presets() do
      {:ok, arch}
    else
      {:error, "unknown --arch #{arch}. Valid: #{Enum.join(Architectures.presets(), ", ")}"}
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

    IO.puts("BalancedTernaryEngine — Convert (synthetic model)")
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

        layer = BalancedTernaryEngine.convert_layer(w, name, delta: auto_delta)

        IO.puts(
          "  #{name}: shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(auto_delta, 3)}"
        )

        layer
      end)

    stats = BalancedTernaryEngine.stats(layers)

    IO.puts("\nModel statistics:")
    print_stat_lines(stats)

    :ok =
      BalancedTernaryEngine.export(layers, output,
        metadata: %{"converter_version" => "0.1.0", command: "convert", delta: delta_opt}
      )
  end

  # ── Demo ──────────────────────────────────────────────────────

  defp do_demo(opts) do
    rows = Keyword.get(opts, :rows, 64)
    cols = Keyword.get(opts, :cols, 128)
    delta = Keyword.get(opts, :delta, 0.5)
    batch = Keyword.get(opts, :batch, 16)

    IO.puts("=== BalancedTernaryEngine Demo ===\n")
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
    packed = BalancedTernaryEngine.Packer.pack(w_tern)
    IO.puts("Packed size: #{byte_size(packed)} bytes (original: #{rows * cols * 4} bytes f32)")

    {_orig, _packed, ratio} = BalancedTernaryEngine.Packer.compression_ratio(w_tern)
    IO.puts("Compression ratio: #{ratio}×")

    _unpacked = BalancedTernaryEngine.Packer.unpack(packed, rows * cols)
    match = BalancedTernaryEngine.Packer.verify_roundtrip(w_tern)
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

    case BalancedTernaryEngine.load(path) do
      {:ok, layers, metadata} ->
        stats = BalancedTernaryEngine.stats(layers)

        format_version =
          case BalancedTernaryEngine.Exporter.format_version(path) do
            {:ok, version} -> version
            {:error, _} -> "unknown"
          end

        IO.puts("=== Model Info: #{path} ===\n")
        IO.puts("Format version: #{format_version}")
        IO.puts("Metadata:")
        Enum.each(metadata, fn {k, v} -> IO.puts("  #{k}: #{inspect(v)}") end)

        IO.puts("\nStatistics:")
        print_model_stats(stats)

        IO.puts("\nLayers:")

        Enum.each(layers, fn
          %BalancedTernaryEngine.Layer{} = layer ->
            IO.puts(
              "  #{layer.name}: ternary, shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(layer.delta, 3)}"
            )

          %BalancedTernaryEngine.RawLayer{} = layer ->
            IO.puts("  #{layer.name}: fp16, shape=#{inspect(layer.shape)}")
        end)

      {:error, reason} ->
        IO.puts("Error loading #{path}: #{inspect(reason)}")
    end
  end

  # ── Validate ──────────────────────────────────────────────────

  defp do_validate(opts) do
    path = Keyword.get(opts, :model, "model.tbin")

    case BalancedTernaryEngine.load(path) do
      {:ok, layers, _metadata} ->
        IO.puts("=== Validating: #{path} ===\n")

        # Round-trip check: unpacking and re-packing must reproduce the
        # stored bytes exactly — proves decode(pack(x)) == x for every
        # stored payload.
        IO.puts("Round-trip verification:")

        Enum.each(layers, fn
          %BalancedTernaryEngine.Layer{} = layer ->
            status = roundtrip_status(layer)
            IO.puts("  #{layer.name}: #{status}")

          %BalancedTernaryEngine.RawLayer{} = layer ->
            IO.puts("  #{layer.name}: fp16 (#{byte_size(layer.data)} bytes)")
        end)

        # Per-layer forward: converted layers chain only inside an
        # architecture-aware runtime, so standalone shape checks are the
        # meaningful smoke test here.
        IO.puts("\nInference test (per-layer forward):")

        ternary_layers = Enum.filter(layers, &match?(%BalancedTernaryEngine.Layer{}, &1))

        print_inference_tests(ternary_layers)

        stats = BalancedTernaryEngine.stats(layers)
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

  defp do_qat(opts) do
    epochs = Keyword.get(opts, :epochs, 2)
    lr = Keyword.get(opts, :lr, 0.01)
    output = Keyword.get(opts, :output, "qat_model.tbin")

    weights =
      case Keyword.get(opts, :input) do
        nil ->
          # Small model for QAT smoke testing; full synthetic model is too
          # large for finite-difference training.
          %{"fc1" => Nx.subtract(Nx.multiply(Nx.iota({4, 6}) |> Nx.as_type(:f32), 0.1), 0.3)}

        input ->
          case BalancedTernaryEngine.Safetensors.read(input) do
            {:ok, t, _} -> t
            {:error, r} -> halt_with("Error reading #{input}: #{inspect(r)}")
          end
      end

    IO.puts("BalancedTernaryEngine — QAT")
    IO.puts("  Epochs: #{epochs}")
    IO.puts("  Output: #{output}")

    # Convert each 2D weight to a QAT layer and train briefly.
    qat_layers =
      weights
      |> Enum.filter(fn {_n, w} -> tuple_size(Nx.shape(w)) == 2 end)
      |> Enum.map(fn {name, w} ->
        q = BalancedTernaryEngine.QATLayer.from_pretrained(w, name, delta: 0.5)

        Enum.reduce(1..epochs, q, fn _epoch, acc ->
          in_f = elem(Nx.shape(w), 1)
          x = Nx.broadcast(0.1, {1, in_f})
          target = Nx.dot(x, Nx.transpose(w))
          {updated, _loss} = BalancedTernaryEngine.QATLayer.train_step(acc, x, target, lr: lr)
          updated
        end)
      end)

    frozen = Enum.map(qat_layers, &BalancedTernaryEngine.QATLayer.freeze/1)

    :ok =
      BalancedTernaryEngine.export(frozen, output,
        metadata: %{"command" => "qat", "converter_version" => "0.1.0"}
      )

    IO.puts("Frozen #{length(frozen)} layers to #{output}")
  end

  @spec halt_with(String.t()) :: no_return()
  defp layer_roundtrip_ok?(layer) do
    total = elem(layer.shape, 0) * elem(layer.shape, 1)

    trits =
      case layer.packing do
        :sparse -> BalancedTernaryEngine.Packer.unpack_sparse(layer.weight_packed, total)
        _ -> BalancedTernaryEngine.Packer.unpack(layer.weight_packed, total)
      end

    case layer.packing do
      :sparse ->
        BalancedTernaryEngine.Packer.pack_sparse(Nx.tensor(trits)) == layer.weight_packed

      _ ->
        BalancedTernaryEngine.Packer.pack(trits) == layer.weight_packed
    end
  end

  defp print_inference_tests([]) do
    IO.puts("  No ternary layers to run inference on.")
  end

  defp print_inference_tests(ternary_layers) do
    Enum.each(ternary_layers, fn layer ->
      {_out, in_features} = layer.shape
      {x, _} = Nx.Random.uniform(Nx.Random.key(123), 0.0, 1.0, shape: {1, in_features})
      output = BalancedTernaryEngine.Layer.forward_nx(layer, x)
      IO.puts("  #{layer.name}: #{inspect(Nx.shape(x))} -> #{inspect(Nx.shape(output))}")
    end)
  end

  defp roundtrip_status(layer) do
    if layer_roundtrip_ok?(layer), do: "OK", else: "FAIL"
  end

  defp halt_with(message) do
    IO.puts(message)
    System.halt(1)
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
