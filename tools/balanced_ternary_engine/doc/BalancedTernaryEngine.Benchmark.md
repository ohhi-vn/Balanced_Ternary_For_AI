# `BalancedTernaryEngine.Benchmark`

Simple benchmarking helpers for conversion and runtime execution.

# `benchmark_conversion`

```elixir
@spec benchmark_conversion(String.t(), String.t(), keyword()) ::
  {:ok, map()} | {:error, term()}
```

# `benchmark_runtime`

```elixir
@spec benchmark_runtime(BalancedTernaryEngine.Runtime.t(), Nx.Tensor.t()) ::
  {:ok, map()}
```

# `measure`

```elixir
@spec measure((-&gt; any())) :: %{time_us: non_neg_integer(), result: any()}
```

---

*Consult [api-reference.md](api-reference.md) for complete listing*
