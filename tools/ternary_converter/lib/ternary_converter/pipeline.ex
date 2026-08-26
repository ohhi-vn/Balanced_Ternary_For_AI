defmodule TernaryConverter.Pipeline do
  @moduledoc """
  End-to-end conversion: safetensors model → `.tbin` balanced ternary model.

      {:ok, summary} =
        TernaryConverter.Pipeline.run("model.safetensors", "model.tbin",
          arch: :llama,
          mode: :absmean
        )

  Steps:

    1. Read all tensors from the source `.safetensors` file
       (`TernaryConverter.Safetensors`)
    2. Classify every tensor via the architecture policy
       (`TernaryConverter.Architectures`) → ternary | keep FP16 | skip
    3. Quantize ternary targets with the selected mode and pack them;
       sensitive tensors are stored as raw FP16 records
    4. Export a version-2 `.tbin` file with provenance metadata

  ## Options

    - `:arch` — `:llama`, `:gpt2`, `:bert`, `:vit`, `:generic`
      (default: `:generic`)
    - `:mode` — quantization mode passed to
      `TernaryConverter.Quantizer.quantize/2`: `:delta`, `:twn`,
      `:absmean` (default: `:twn`)
    - `:delta` — explicit threshold when mode is `:delta`
    - `:compress` — gzip the output (default: false)
  """

  alias TernaryConverter.{Architectures, Exporter, Layer, RawLayer, Safetensors, Stats}

  @spec run(String.t(), String.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def run(input_path, output_path, opts \\ []) do
    arch = Keyword.get(opts, :arch, :generic)
    compress = Keyword.get(opts, :compress, false)

    unless arch in Architectures.presets() do
      raise ArgumentError,
            "unknown architecture #{inspect(arch)}; expected one of #{inspect(Architectures.presets())}"
    end

    with {:ok, tensors, source_metadata} <- Safetensors.read(input_path) do
      {ternary_entries, fp16_entries} = partition(tensors, arch)

      layers =
        Enum.map(ternary_entries, fn {name, w} ->
          Layer.from_dense(w, name, Keyword.take(opts, [:mode, :delta]))
        end)

      raw_layers =
        Enum.map(fp16_entries, fn {name, w} ->
          RawLayer.from_tensor(w, name)
        end)

      all_layers = Enum.sort_by(layers ++ raw_layers, & &1.name)

      stats = Stats.compute(layers)

      metadata =
        %{
          "source_file" => Path.basename(input_path),
          "architecture" => to_string(arch),
          "mode" => to_string(Keyword.get(opts, :mode, :twn)),
          "source_tensors" => map_size(tensors),
          "ternary_layers" => length(layers),
          "fp16_layers" => length(raw_layers),
          "skipped_tensors" => map_size(tensors) - length(layers) - length(raw_layers),
          "source_metadata" => source_metadata,
          "converted_at" => DateTime.utc_now() |> DateTime.to_iso8601()
        }

      :ok = Exporter.export(all_layers, output_path, metadata: metadata, compress: compress)

      {:ok,
       %{
         input_path: input_path,
         output_path: output_path,
         arch: arch,
         ternary_layers: layers,
         fp16_layers: raw_layers,
         skipped: map_size(tensors) - length(layers) - length(raw_layers),
         stats: stats,
         metadata: metadata
       }}
    end
  end

  defp partition(tensors, arch) do
    Enum.reduce(tensors, {[], []}, fn {name, tensor}, {ternary, fp16} ->
      case Architectures.classify(arch, name) do
        :ternary ->
          {[{name, Architectures.transform(arch, name, tensor)} | ternary], fp16}

        :keep_fp16 ->
          {ternary, [{name, Architectures.transform(arch, name, tensor)} | fp16]}

        :skip ->
          {ternary, fp16}
      end
    end)
  end
end
