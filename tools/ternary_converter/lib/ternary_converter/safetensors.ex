defmodule TernaryConverter.Safetensors do
  @moduledoc """
  Pure-Elixir reader/writer for the HuggingFace `.safetensors` format.

  Layout:

      [header_len: u64 little-endian][header: UTF-8 JSON][tensor data]

  Header JSON maps tensor names to `%{"dtype" => ..., "shape" => [...],
  "data_offsets" => [begin, end]}` where offsets are relative to the
  start of the data section. An optional `__metadata__` key holds a
  string-to-string map.

  Float tensors are returned as `:f32`; integer tensors as `:s64`.
  """

  @dtype_to_nx %{
    "F64" => {:f, 64},
    "F32" => {:f, 32},
    "F16" => {:f, 16},
    "BF16" => {:bf, 16},
    "I64" => {:s, 64},
    "I32" => {:s, 32},
    "I16" => {:s, 16},
    "I8" => {:s, 8},
    "U8" => {:u, 8},
    "BOOL" => {:u, 8}
  }

  @nx_to_dtype %{
    {:f, 64} => "F64",
    {:f, 32} => "F32",
    {:f, 16} => "F16",
    {:bf, 16} => "BF16",
    {:s, 64} => "I64",
    {:s, 32} => "I32",
    {:s, 16} => "I16",
    {:s, 8} => "I8",
    {:u, 8} => "U8"
  }

  # ── Reading ──────────────────────────────────────────────────

  @doc """
  Read a safetensors file.

  Returns `{:ok, tensors, metadata}` where `tensors` is
  `%{name => Nx.Tensor.t()}` (floats converted to f32) and `metadata`
  is the optional `__metadata__` map (empty map if absent).
  """
  @spec read(String.t() | binary()) ::
          {:ok, %{String.t() => Nx.Tensor.t()}, %{String.t() => String.t()}}
          | {:error, term()}
  def read(path) when is_binary(path) do
    case File.read(path) do
      {:ok, binary} -> decode(binary)
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Decode safetensors from a binary.

  Any structural problem (truncated file, bad JSON, bad offsets) returns
  `{:error, :invalid_safetensors}`; an unknown dtype raises `ArgumentError`
  since it indicates a caller/producer bug rather than corrupt input.
  """
  @spec decode(binary()) :: {:ok, map(), map()} | {:error, term()}
  def decode(binary) when is_binary(binary) do
    with <<header_len::64-little, rest::binary>> when byte_size(rest) >= header_len <- binary,
         <<header_json::binary-size(header_len), data::binary>> <- rest,
         {:ok, header} <- Jason.decode(header_json) do
      {entries, metadata} = split_header(header)

      tensors =
        Map.new(entries, fn {name, info} ->
          {name, build_tensor(info, data)}
        end)

      {:ok, tensors, metadata}
    else
      _ -> {:error, :invalid_safetensors}
    end
  end

  def decode(_other), do: {:error, :invalid_safetensors}

  defp split_header(header) do
    # "__metadata__" is an optional header key reserved for arbitrary
    # string metadata; everything else must be a tensor entry.
    case Map.pop(header, "__metadata__") do
      {nil, entries} -> {entries, %{}}
      {metadata, entries} -> {entries, metadata}
    end
  end

  defp build_tensor(%{"dtype" => dtype, "shape" => shape, "data_offsets" => [begin_, end_]}, data)
       when is_map_key(@dtype_to_nx, dtype) do
    nx_type = Map.fetch!(@dtype_to_nx, dtype)

    data
    |> binary_part(begin_, end_ - begin_)
    |> Nx.from_binary(nx_type)
    |> normalize_type()
    |> Nx.reshape(List.to_tuple(shape))
  end

  defp build_tensor(%{"dtype" => dtype}, _data) do
    raise ArgumentError, "unsupported safetensors dtype: #{inspect(dtype)}"
  end

  # Upcast on load so downstream code sees exactly two types:
  # f32 for floats (uniform math, no mixed-precision surprises) and
  # s64 for index-like integers.
  defp normalize_type(t) do
    cond do
      Nx.Type.float?(Nx.type(t)) -> Nx.as_type(t, :f32)
      true -> Nx.as_type(t, :s64)
    end
  end

  # ── Writing ──────────────────────────────────────────────────

  @doc """
  Encode a map of tensors to safetensors binary.

  Supported tensor types are written as-is; anything else is converted
  to f32. Data blocks are padded to 8-byte alignment.
  """
  @spec encode(%{String.t() => Nx.Tensor.t()}, %{String.t() => String.t()}) :: binary()
  def encode(tensors, metadata \\ %{}) do
    {entries_rev, blobs_rev, _offset} =
      Enum.reduce(tensors, {[], [], 0}, fn {name, tensor}, {entries, blobs, offset} ->
        {dtype_name, blob} = to_dtype_and_binary(tensor)
        size = byte_size(blob)
        pad = rem(8 - rem(size, 8), 8)

        entry =
          {name,
           %{
             "dtype" => dtype_name,
             "shape" => Nx.shape(tensor) |> Tuple.to_list(),
             "data_offsets" => [offset, offset + size]
           }}

        {
          [entry | entries],
          [pad_bytes(pad), blob | blobs],
          offset + size + pad
        }
      end)

    header = Map.put(Map.new(Enum.reverse(entries_rev)), "__metadata__", metadata)
    header_json = Jason.encode!(header)

    # Pad the header itself so the data section starts 8-byte aligned.
    pad = rem(8 - rem(byte_size(header_json), 8), 8)
    header_padded = header_json <> pad_bytes(pad)

    <<
      byte_size(header_padded)::64-little,
      header_padded::binary,
      IO.iodata_to_binary(Enum.reverse(blobs_rev))::binary
    >>
  end

  # Spaces, not NULs — the header length includes the padding and JSON
  # parsers tolerate trailing whitespace only.
  defp pad_bytes(0), do: ""
  defp pad_bytes(n), do: String.duplicate(" ", n)

  defp to_dtype_and_binary(tensor) do
    type = Nx.type(tensor)

    if Map.has_key?(@nx_to_dtype, type) do
      {Map.fetch!(@nx_to_dtype, type), Nx.to_binary(tensor)}
    else
      {"F32", tensor |> Nx.as_type(:f32) |> Nx.to_binary()}
    end
  end

  @doc """
  Write a map of tensors to a `.safetensors` file.
  """
  @spec write(String.t(), %{String.t() => Nx.Tensor.t()}, %{String.t() => String.t()}) ::
          :ok | {:error, term()}
  def write(path, tensors, metadata \\ %{}) do
    File.write(path, encode(tensors, metadata))
  end
end
