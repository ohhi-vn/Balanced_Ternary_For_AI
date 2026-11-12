# `BalancedTernaryEngine.Runtime`

Minimal runtime for executing a converted `.tbin` model.

Supports layer chaining, softmax, layer normalization, multi-head
attention with a KV cache, and prefill/decode phases.

# `t`

```elixir
@type t() :: %BalancedTernaryEngine.Runtime{
  cache: map(),
  config: map(),
  layers: [any()]
}
```

# `attention`

```elixir
@spec attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map(), integer()) ::
  {Nx.Tensor.t(), map()}
```

Multi-head attention with a KV cache.

`q`, `k`, `v` are `{seq, hidden}` tensors. Returns `{out, new_cache}`.

# `decode`

```elixir
@spec decode(t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), t()}
```

Decode: run a single new token, reusing the cache.

# `forward`

```elixir
@spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
```

Run the model (layer chaining) on an input tensor.

# `layer_norm`

```elixir
@spec layer_norm(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
```

Layer normalization (RMSNorm-style) over the last axis.

# `load`

```elixir
@spec load(String.t(), map()) :: {:ok, t()} | {:error, term()}
```

Load a `.tbin` model and an optional config map.

# `new`

```elixir
@spec new([any()], map()) :: {:ok, t()} | {:error, term()}
```

Create a runtime from already-loaded layers and a config map.

# `prefill`

```elixir
@spec prefill(t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), t()}
```

Prefill: run a full sequence through the model, populating the cache.

# `reset_cache`

```elixir
@spec reset_cache(t()) :: t()
```

Reset the runtime cache without reloading the model.

# `softmax`

```elixir
@spec softmax(Nx.Tensor.t()) :: Nx.Tensor.t()
```

Softmax over the last axis.

# `validate`

```elixir
@spec validate(t()) :: :ok | {:error, term()}
```

Validate a runtime instance.

# `validate_config`

```elixir
@spec validate_config(term()) :: {:ok, map()} | {:error, term()}
```

Validate a runtime configuration map.

Returns `{:ok, config}` when every supplied value is valid, or
`{:error, reason}` otherwise. Unknown keys are allowed so callers can
carry application-specific settings.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
