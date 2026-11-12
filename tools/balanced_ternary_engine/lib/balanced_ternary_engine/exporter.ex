defmodule BalancedTernaryEngine.Exporter do
  @moduledoc """
  Exports and imports ternary models to/from .tbin binary format.

  The .tbin format:

      [Header][Record 1][Record 2]...[Record N]

      Header:
        magic:       4 bytes   ("TBN\\0")
        version:     4 bytes   (uint32; 2 as of mixed-precision support)
        num_layers:  4 bytes   (uint32)
        meta_len:    4 bytes   (uint32)
        metadata:    meta_len bytes (UTF-8 JSON)

      Record (version >= 2):
        kind:        1 byte    (0 = ternary, 1 = raw FP16)
        size:        4 bytes   (uint32)
        payload:     size bytes
          - ternary: BalancedTernaryEngine.Layer.to_binary/1
          - raw:     BalancedTernaryEngine.RawLayer.to_binary/1

      Record (version 1):
        size:        4 bytes   (uint32)
        payload:     size bytes (ternary layer only)

  Version 1 files remain readable; they are written as version 2 on the
  next export.
  """

  @tbin_magic "TBN\0"
  @tbin_version 2

  @kind_ternary 0
  @kind_raw_fp16 1

  @doc """
  Export a list of layers to a .tbin file.

  Layers may be `BalancedTernaryEngine.Layer` (ternary) or
  `BalancedTernaryEngine.RawLayer` (kept at FP16); each is tagged with its
  record kind.

  ## Options

    - `:metadata` — Map of metadata (model name, version, etc.)
    - `:compress` — Whether to compress with gzip (default: false)
  """
  @spec export([struct()], String.t(), keyword()) :: :ok | {:error, term()}
  def export(layers, output_path, opts \\ []) do
    metadata = Keyword.get(opts, :metadata, %{})
    compress = Keyword.get(opts, :compress, false)

    binary = build_tbin(layers, metadata, compress)

    case File.write(output_path, binary) do
      :ok ->
        size_mb = byte_size(binary) / 1024 / 1024

        IO.puts(
          "Exported #{length(layers)} layers to #{output_path} (#{Float.round(size_mb, 1)} MB)"
        )

        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_tbin(layers, metadata, compress) do
    metadata_binary = Jason.encode!(metadata)

    records =
      Enum.map(layers, fn
        %BalancedTernaryEngine.Layer{} = layer ->
          payload = BalancedTernaryEngine.Layer.to_binary(layer)
          <<@kind_ternary::8, byte_size(payload)::32, payload::binary>>

        %BalancedTernaryEngine.RawLayer{} = layer ->
          payload = BalancedTernaryEngine.RawLayer.to_binary(layer)
          <<@kind_raw_fp16::8, byte_size(payload)::32, payload::binary>>
      end)

    binary = <<
      @tbin_magic::binary,
      @tbin_version::32,
      length(layers)::32,
      byte_size(metadata_binary)::32,
      metadata_binary::binary,
      IO.iodata_to_binary(records)::binary
    >>

    if compress, do: :zlib.gzip(binary), else: binary
  end

  @doc """
  Load a .tbin file back into a list of layers.
  """
  @spec load(String.t()) ::
          {:ok, [struct()], map()} | {:error, term()}
  def load(input_path) do
    with {:ok, binary} <- File.read(input_path),
         binary <- maybe_decompress(binary),
         {:ok, layers, metadata} <- parse_tbin(binary) do
      {:ok, layers, metadata}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Return the format version stored in a `.tbin` file.
  """
  @spec format_version(String.t()) :: {:ok, non_neg_integer()} | {:error, term()}
  def format_version(input_path) do
    with {:ok, binary} <- File.read(input_path),
         binary <- maybe_decompress(binary),
         <<@tbin_magic::binary, version::32, _rest::binary>> <- binary do
      {:ok, version}
    else
      {:error, reason} -> {:error, reason}
      _ -> {:error, "invalid .tbin format"}
    end
  end

  defp maybe_decompress(binary) do
    case binary do
      <<31, 139, _rest::binary>> -> :zlib.gunzip(binary)
      _ -> binary
    end
  end

  @doc """
  Validate that a `.tbin` file can be loaded.
  """
  @spec validate(String.t()) :: :ok | {:error, term()}
  def validate(path) do
    case load(path) do
      {:ok, _layers, _meta} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp parse_tbin(<<
         @tbin_magic::binary,
         version::32,
         num_layers::32,
         meta_len::32,
         metadata_binary::binary-size(meta_len),
         layers_binary::binary
       >>)
       when version in [1, 2] do
    metadata = Jason.decode!(metadata_binary)
    layers = parse_records(layers_binary, num_layers, version, [])
    {:ok, Enum.reverse(layers), metadata}
  rescue
    e -> {:error, Exception.message(e)}
  end

  defp parse_tbin(_), do: {:error, "invalid .tbin format"}

  defp parse_records(<<>>, 0, _version, acc), do: acc

  defp parse_records(binary, remaining, version, acc) do
    {layer, rest} =
      case version do
        1 ->
          <<size::32, payload::binary-size(size), rest::binary>> = binary
          {BalancedTernaryEngine.Layer.from_binary(payload), rest}

        2 ->
          <<kind::8, size::32, payload::binary-size(size), rest::binary>> = binary

          layer =
            case kind do
              @kind_ternary -> BalancedTernaryEngine.Layer.from_binary(payload)
              @kind_raw_fp16 -> elem(BalancedTernaryEngine.RawLayer.from_binary(payload), 0)
            end

          {layer, rest}
      end

    parse_records(rest, remaining - 1, version, [layer | acc])
  end
end
