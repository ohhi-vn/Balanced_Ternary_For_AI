# `BalancedTernaryEngine.Packer`

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

# `compression_ratio`

```elixir
@spec compression_ratio(Nx.Tensor.t()) ::
  {non_neg_integer(), non_neg_integer(), float()}
```

Compute compression ratio for a given tensor.

Returns `{original_bytes, packed_bytes, ratio}`.

# `pack`

```elixir
@spec pack(Nx.Tensor.t() | [integer()]) :: binary()
```

Pack a ternary tensor into a binary (10 trits per 16-bit word).

## Examples

    iex> t = Nx.tensor([1, 0, -1, 1, 1, 0, -1, 0, 1, -1])
    iex> BalancedTernaryEngine.Packer.pack(t)
    <<14285::16-native>>

# `pack_sparse`

```elixir
@spec pack_sparse(Nx.Tensor.t()) :: binary()
```

Pack with sparse encoding for high-sparsity layers.

Stores only non-zero entries as {index, sign} pairs.
More efficient than dense packing when sparsity > ~88%.

Format per entry (16 bits):
  - Index: 15 bits (position in flattened tensor, max 32767)
  - Sign:  1 bit  (0 for +1, 1 for -1)

A u32 count prefixes the entry list so `unpack_sparse/2` knows how
many entries to read without relying on binary length alone.

# `unpack`

```elixir
@spec unpack(binary(), non_neg_integer()) :: [integer()]
```

Unpack a binary back into a flat list of trits.

## Examples

    iex> BalancedTernaryEngine.Packer.unpack(<<14285::16-native>>, 10)
    [1, 0, -1, 1, 1, 0, -1, 0, 1, -1]

# `unpack_sparse`

```elixir
@spec unpack_sparse(binary(), non_neg_integer()) :: [integer()]
```

Unpack a sparse-encoded binary.

# `verify_roundtrip`

```elixir
@spec verify_roundtrip(Nx.Tensor.t()) :: boolean()
```

Verify round-trip: pack then unpack should return the original.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
