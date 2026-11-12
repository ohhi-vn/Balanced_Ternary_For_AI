# `BalancedTernaryEngine.Validation`

Reusable validation helpers for tensors, configs, and model files.

# `validate_config`

```elixir
@spec validate_config(term()) :: {:ok, map()} | {:error, term()}
```

# `validate_model_file`

```elixir
@spec validate_model_file(String.t()) :: :ok | {:error, term()}
```

# `validate_tensor`

```elixir
@spec validate_tensor(term()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
```

---

*Consult [api-reference.md](api-reference.md) for complete listing*
