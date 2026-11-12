# `BalancedTernaryEngine.Layer`

A neural network layer with ternary weights.

Supports:
- Forward pass with add/sub/skip (no multiplication)
- Per-channel scale factors and bias
- Conversion from dense (FP32) weights
- Packed weight storage (10 trits per 16-bit word)
- Serialization to/from binary format

# `t`

```elixir
@type t() :: %BalancedTernaryEngine.Layer{
  bias: [float()],
  delta: float(),
  name: String.t(),
  original_shape: term(),
  packing: term(),
  scales: [float()],
  shape: {non_neg_integer(), non_neg_integer()},
  sparsity: float(),
  weight_packed: binary()
}
```

# `forward`

```elixir
@spec forward(t(), [number()]) :: [number()]
```

Forward pass through the ternary layer.

Computes: output[i] = scale[i] * sum_j(W_ternary[i][j] * x[j]) + bias[i]

Uses add/sub/skip — no multiplication for the weight-activation product.

## Examples

    iex> w = Nx.tensor([[0.8, -0.3, 0.1], [0.5, -0.2, 0.7]])
    iex> layer = BalancedTernaryEngine.Layer.from_dense(w, "test", delta: 0.4)
    iex> BalancedTernaryEngine.Layer.forward(layer, [1.0, 2.0, 3.0])
    [0.5666666626930237, 1.619999885559082]

# `forward_nx`

```elixir
@spec forward_nx(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
```

Forward pass with Nx tensors (for batched inference).

# `from_binary`

```elixir
@spec from_binary(binary()) :: t()
```

Deserialize layer from binary.

Mirrors `to_binary/1` field-for-field. The pattern match is total:
a truncated or malformed payload fails to match and raises, which the
Exporter wraps into an error tuple.

# `from_dense`

```elixir
@spec from_dense(Nx.Tensor.t(), String.t(), keyword()) :: t()
```

Create a TernaryLayer from a dense (FP32) weight matrix.

Quantization is delegated to `BalancedTernaryEngine.Quantizer.quantize/2`;
pass `mode:` (`:delta`, `:twn`, `:absmean`) and `delta:` options
through. Automatically packs weights and records per-channel scales.

## Examples

    iex> w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
    iex> layer = BalancedTernaryEngine.Layer.from_dense(w, "fc1", delta: 0.4)
    iex> layer.shape
    {2, 4}
    iex> layer.name
    "fc1"

# `to_binary`

```elixir
@spec to_binary(t()) :: binary()
```

Serialize layer to binary (.tbin record payload).

Layout (all integers big-endian):

    name_len::16 | name | out::32 | in::32
    delta::float-32 | sparsity::float-32
    num_scales::32 | scales (num_scales × float-16)
    num_biases::32 | biases  (num_biases × float-16)
    packed_size::32 | packed trits (see Packer)

Scales/biases use half precision: they are calibration constants, and
f16 halves their footprint at negligible accuracy cost.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
