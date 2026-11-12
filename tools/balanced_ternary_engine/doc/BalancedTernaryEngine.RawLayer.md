# `BalancedTernaryEngine.RawLayer`

A dense weight matrix kept at half precision inside a converted model.

Quantization-sensitive tensors — token embeddings, LM head, layer
norms — are stored as `RawLayer` instead of being ternarized
(see `BalancedTernaryEngine.Architectures`). Weights are held as raw
float16 bytes; activations are computed in f32.

# `t`

```elixir
@type t() :: %BalancedTernaryEngine.RawLayer{
  data: binary(),
  name: String.t(),
  shape: tuple()
}
```

# `forward_nx`

```elixir
@spec forward_nx(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
```

Forward pass: dequantize to f32 and apply as a linear projection.

Handles both `(out, in)` (`y = x·Wᵀ`) and `(in, out)` (`y = x·W`)
layouts — GPT-2-style checkpoints store Conv1D weights transposed, and
the converter preserves whichever layout it was given. Intended for
lm_head-style tensors; embeddings and norm vectors are consumed by
runtimes via `to_tensor/1`.

# `from_binary`

```elixir
@spec from_binary(binary()) :: {t(), binary()}
```

Deserialize from binary. Returns `{layer, rest_binary}`.

# `from_tensor`

```elixir
@spec from_tensor(Nx.Tensor.t(), String.t()) :: t()
```

Build a RawLayer from a tensor of any float type (stored as f16).

# `packed_bytes`

```elixir
@spec packed_bytes(t()) :: non_neg_integer()
```

Byte size of the stored f16 payload (for stats).

# `to_binary`

```elixir
@spec to_binary(t()) :: binary()
```

Serialize to binary: name, shape, f16 payload.

# `to_tensor`

```elixir
@spec to_tensor(t()) :: Nx.Tensor.t()
```

Materialize the weights as an f32 tensor.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
