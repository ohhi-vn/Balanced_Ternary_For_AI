defmodule BalancedTernaryEngine.Packer do
  @moduledoc """
  Packs ternary {-1, 0, +1} tensors into compact binary format.

  Supports:
  - Dense packing: 10 trits per 16-bit word (base-3 encoding)
  - Sparse encoding: index + sign for high-sparsity layers
  - Round-trip verification
  - Compression ratio calculation

  ## Dense word layout

  Each group of 10 trits is shifted to base-3 digits via `trit + 1`
  ({-1,0,+1} → {0,1,2}) and read **most-significant digit first**, so a
  group's first weight occupies the highest-value position:

      word = d₀·3⁹ + d₁·3⁸ + … + d₉·3⁰     where dᵢ = tritᵢ + 1

  Since 3¹⁰ = 59_049 < 2¹⁶ = 65_536 every word fits in 16 bits (~90%
  efficiency). Trailing weights of the final group are zero-padded.

  ## Sparse layout

  `<<count::32>>` followed by `count` entries of `<<index::15, sign::1>>`
  (sign: 0 → +1, 1 → −1). Only wins above ~88–90% sparsity because an
  entry costs 16 bits versus 1.6 bits per densely packed weight.
  """

  @trits_per_word 10

  # ── Dense Packing ────────────────────────────────────────────

  @doc """
  Pack a ternary tensor into a binary (10 trits per 16-bit word).

  ## Examples

      iex> t = Nx.tensor([1, 0, -1, 1, 1, 0, -1, 0, 1, -1])
      iex> BalancedTernaryEngine.Packer.pack(t)
      <<14285::16-native>>
  """
  @spec pack(Nx.Tensor.t() | [integer()]) :: binary()
  def pack(tensor) when is_struct(tensor, Nx.Tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&round/1)
    |> pack_trits()
  end

  def pack(trits) when is_list(trits) do
    pack_trits(Enum.map(trits, &round/1))
  end

  defp pack_trits(trits) do
    trits
    |> Enum.chunk_every(@trits_per_word)
    |> Enum.map(fn chunk ->
      padded = pad_chunk(chunk, @trits_per_word)
      encode_word(padded)
    end)
    |> :erlang.list_to_binary()
  end

  defp pad_chunk(chunk, size) do
    chunk ++ List.duplicate(0, size - length(chunk))
  end

  defp encode_word(trits) do
    trits
    |> Enum.map(&(&1 + 1))
    # Reverse so the group's first trit becomes the most significant
    # digit of the base-3 number — this makes unpack restore original
    # order without a final reverse (see module doc for the layout).
    |> Enum.reverse()
    |> Enum.reduce(0, fn digit, acc -> acc * 3 + digit end)
    |> then(&<<&1::16-integer-native>>)
  end

  @doc """
  Unpack a binary back into a flat list of trits.

  ## Examples

      iex> BalancedTernaryEngine.Packer.unpack(<<14285::16-native>>, 10)
      [1, 0, -1, 1, 1, 0, -1, 0, 1, -1]
  """
  @spec unpack(binary(), non_neg_integer()) :: [integer()]
  def unpack(binary, total_trits) do
    words = for <<word::16-integer-native <- binary>>, do: word

    words
    |> Enum.flat_map(&unpack_word/1)
    |> Enum.take(total_trits)
  end

  defp unpack_word(word) do
    word
    |> extract_digits(@trits_per_word, [])
    |> Enum.map(fn
      0 -> -1
      1 -> 0
      2 -> 1
    end)
  end

  defp extract_digits(_v, 0, acc), do: Enum.reverse(acc)

  # Peels least-significant base-3 digits one at a time; prepending each
  # to `acc` leaves the result most-significant-first, matching the pack
  # order (first weight = highest digit).
  defp extract_digits(v, n, acc) do
    extract_digits(div(v, 3), n - 1, [rem(v, 3) | acc])
  end

  # ── Sparse Packing ───────────────────────────────────────────

  @doc """
  Pack with sparse encoding for high-sparsity layers.

  Stores only non-zero entries as {index, sign} pairs.
  More efficient than dense packing when sparsity > ~88%.

  Format per entry (16 bits):
    - Index: 15 bits (position in flattened tensor, max 32767)
    - Sign:  1 bit  (0 for +1, 1 for -1)

  A u32 count prefixes the entry list so `unpack_sparse/2` knows how
  many entries to read without relying on binary length alone.
  """
  @spec pack_sparse(Nx.Tensor.t()) :: binary()
  def pack_sparse(tensor) do
    flat = Nx.to_flat_list(tensor) |> Enum.map(&round/1)

    entries =
      flat
      |> Enum.with_index()
      |> Enum.filter(fn {v, _i} -> v != 0 end)
      |> Enum.map(fn {v, i} ->
        sign = if v == 1, do: 0, else: 1
        <<i::15, sign::1>>
      end)

    <<length(entries)::32>> <> :erlang.list_to_binary(entries)
  end

  @doc """
  Unpack a sparse-encoded binary.
  """
  @spec unpack_sparse(binary(), non_neg_integer()) :: [integer()]
  def unpack_sparse(binary, total_trits) do
    <<_num_entries::32, entries_binary::binary>> = binary

    indices_and_signs =
      for <<index::15, sign::1 <- entries_binary>>, do: {index, sign}

    result = List.duplicate(0, total_trits)

    Enum.reduce(indices_and_signs, result, fn {idx, sign}, acc ->
      value = if sign == 0, do: 1, else: -1
      List.replace_at(acc, idx, value)
    end)
  end

  # ── Utilities ────────────────────────────────────────────────

  @doc """
  Verify round-trip: pack then unpack should return the original.
  """
  @spec verify_roundtrip(Nx.Tensor.t()) :: boolean()
  def verify_roundtrip(tensor) do
    flat = Nx.to_flat_list(tensor) |> Enum.map(&round/1)
    total = length(flat)
    packed = pack(tensor)
    unpacked = unpack(packed, total)
    flat == unpacked
  end

  @doc """
  Compute compression ratio for a given tensor.

  Returns `{original_bytes, packed_bytes, ratio}`.
  """
  @spec compression_ratio(Nx.Tensor.t()) :: {non_neg_integer(), non_neg_integer(), float()}
  def compression_ratio(tensor) do
    total_trits = Nx.size(tensor) |> Nx.to_number()
    original_bytes = total_trits * 4

    packed = pack(tensor)
    packed_bytes = byte_size(packed)

    ratio = original_bytes / max(packed_bytes, 1)
    {original_bytes, packed_bytes, Float.round(ratio, 2)}
  end
end
