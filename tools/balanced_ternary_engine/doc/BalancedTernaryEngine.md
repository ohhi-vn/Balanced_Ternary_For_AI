# `BalancedTernaryEngine`

Main API for converting pretrained models to balanced ternary format.

## Quick Start

    # Create a ternary layer from random weights
    w = Nx.random_uniform({64, 128})
    layer = BalancedTernaryEngine.convert_layer(w, "fc1", delta: 0.5)

    # Run inference
    input = Nx.random_uniform({1, 128})
    output = BalancedTernaryEngine.inference([layer], input)

    # Export to .tbin
    :ok = BalancedTernaryEngine.export([layer], "model.tbin")

    # Load back
    {:ok, layers, metadata} = BalancedTernaryEngine.load("model.tbin")

## Full Pipeline

    # 1. Create synthetic model weights
    weights = %{
      "fc1" => Nx.random_uniform({256, 512}),
      "fc2" => Nx.random_uniform({128, 256}),
      "fc3" => Nx.random_uniform({64, 128})
    }

    # 2. Analyze sensitivity
    sample_input = Nx.random_uniform({1, 512})
    sens = BalancedTernaryEngine.analyze_sensitivity(weights, sample_input)

    # 3. Convert all layers
    layers = BalancedTernaryEngine.convert_all(weights, delta: 0.5)

    # 4. Validate
    BalancedTernaryEngine.validate(layers, weights, sample_input)

    # 5. Export
    :ok = BalancedTernaryEngine.export(layers, "model.tbin")

# `analyze_sensitivity`

```elixir
@spec analyze_sensitivity(
  %{required(String.t()) =&gt; Nx.Tensor.t()},
  Nx.Tensor.t(),
  float()
) :: [
  {String.t(), float()}
]
```

Analyze per-layer sensitivity.

# `auto_mixed_precision`

```elixir
@spec auto_mixed_precision(
  %{required(String.t()) =&gt; Nx.Tensor.t()},
  Nx.Tensor.t(),
  float(),
  float()
) ::
  {%{required(String.t()) =&gt; :ternary | :fp16}, non_neg_integer(),
   non_neg_integer()}
```

Automatically determine mixed-precision assignment.

# `convert_all`

```elixir
@spec convert_all(%{required(String.t()) =&gt; Nx.Tensor.t()}, keyword()) :: [
  BalancedTernaryEngine.Layer.t()
]
```

Convert all weight tensors to ternary layers.

Accepts the same quantization options as `Layer.from_dense/3`
(`mode:` / `delta:` — see `Quantizer.quantize/2`). The result is
sorted by layer name for deterministic file layout.

# `convert_layer`

```elixir
@spec convert_layer(Nx.Tensor.t(), String.t(), keyword()) ::
  BalancedTernaryEngine.Layer.t()
```

Convert a single weight matrix to a ternary layer.

# `export`

```elixir
@spec export([BalancedTernaryEngine.Layer.t()], String.t(), keyword()) ::
  :ok | {:error, term()}
```

Export layers to .tbin file.

# `inference`

```elixir
@spec inference([BalancedTernaryEngine.Layer.t()], Nx.Tensor.t()) :: Nx.Tensor.t()
```

Run inference through a list of ternary layers.

# `load`

```elixir
@spec load(String.t()) ::
  {:ok, [BalancedTernaryEngine.Layer.t()], map()} | {:error, term()}
```

Load layers from .tbin file.

# `stats`

```elixir
@spec stats([BalancedTernaryEngine.Layer.t()]) :: map()
```

Get model statistics.

# `validate`

```elixir
@spec validate(
  [BalancedTernaryEngine.Layer.t()],
  %{required(String.t()) =&gt; Nx.Tensor.t()},
  Nx.Tensor.t()
) :: map()
```

Run validation: sparsity, similarity, inference test, round-trip.

Prints a human-readable report and returns the computed metrics as a
map for programmatic use. Note: this function performs I/O; call the
underlying pieces (`stats/1`, `inference/2`, `Packer`) directly when a
pure result is needed.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
