# `BalancedTernaryEngine.QATLayer`

Quantization-Aware Training layer with ternary weights.

Maintains FP32 shadow weights that are ternarized on each forward pass.
Uses a straight-through estimator (STE) so gradients flow to the shadow
weights during training: the forward pass uses ternary weights, while
the gradient is applied to the shadow weights directly.

This module provides the primitives (ternarize, forward, update, freeze).
The training loop composes these with a loss and an optimizer step.

# `t`

```elixir
@type t() :: %BalancedTernaryEngine.QATLayer{
  bias: Nx.Tensor.t(),
  delta: float(),
  name: String.t(),
  scale: Nx.Tensor.t(),
  weight_shadow: Nx.Tensor.t()
}
```

# `forward`

```elixir
@spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
```

Forward pass: output = (ternary_weight @ input) + bias.

# `freeze`

```elixir
@spec freeze(t()) :: BalancedTernaryEngine.Layer.t()
```

Freeze a QAT layer into a standard BalancedTernaryEngine.Layer for inference.

# `from_pretrained`

```elixir
@spec from_pretrained(Nx.Tensor.t(), String.t(), keyword()) :: t()
```

Create a QAT layer from a pretrained weight tensor.

# `ternarize`

```elixir
@spec ternarize(t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
```

Ternarize shadow weights. Returns `{effective, shadow}` where `effective`
is the ternary (scaled) weight used in the forward pass and `shadow` is
the FP32 shadow weight retained for gradient updates (STE).

# `train_step`

```elixir
@spec train_step(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: {t(), float()}
```

Apply a straight-through training step using a finite-difference
gradient of the supplied loss function.

`loss_fn` receives the layer and returns a scalar loss. The gradient
w.r.t. the shadow weights is approximated and used to update them.
Returns the updated layer and the loss value.

# `update`

```elixir
@spec update(t(), Nx.Tensor.t(), float()) :: t()
```

SGD update of shadow weights: w_shadow = w_shadow - lr * grad.

# `validate_frozen`

```elixir
@spec validate_frozen(t()) ::
  {:ok, BalancedTernaryEngine.Layer.t()} | {:error, term()}
```

Freeze and validate a QAT layer.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
