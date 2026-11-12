# `BalancedTernaryEngine.Sensitivity`

Analyzes per-layer sensitivity to ternary quantization.

Identifies which layers can be safely ternarized and which should
remain at higher precision (INT8/FP16).

# `analyze`

```elixir
@spec analyze([{String.t(), Nx.Tensor.t()}], Nx.Tensor.t(), float()) :: [
  {String.t(), float()}
]
```

Analyze sensitivity of each quantizable layer.

For each layer:
  1. Measure output with FP32 weights (reference)
  2. Measure output with ternary weights
  3. Compute MSE between outputs

Returns a list of `{layer_name, mse}` sorted by sensitivity
(highest first = most sensitive = keep FP16).

# `auto_mixed_precision`

```elixir
@spec auto_mixed_precision(
  [{String.t(), Nx.Tensor.t()}],
  Nx.Tensor.t(),
  float(),
  float()
) ::
  {%{required(String.t()) =&gt; :ternary | :fp16}, non_neg_integer(),
   non_neg_integer()}
```

Automatically determine mixed-precision assignment.

Starts with all layers ternary, then greedily promotes the most
sensitive layer to FP16 until the accuracy target is met.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
