# `BalancedTernaryEngine.Stats`

Aggregate statistics over a list of model layers.

Accepts mixed lists of `BalancedTernaryEngine.Layer` (ternary) and
`BalancedTernaryEngine.RawLayer` (FP16) records. Size/sparsity metrics are
computed over the ternary subset; FP16 records contribute their byte
size to the packed total.

# `compute`

```elixir
@spec compute([struct()]) :: %{
  num_layers: non_neg_integer(),
  ternary_layers: non_neg_integer(),
  fp16_layers: non_neg_integer(),
  total_parameters: non_neg_integer(),
  nonzero_parameters: non_neg_integer(),
  zero_parameters: non_neg_integer(),
  overall_sparsity: float(),
  original_size_mb: float(),
  packed_size_mb: float(),
  compression_ratio: float()
}
```

---

*Consult [api-reference.md](api-reference.md) for complete listing*
