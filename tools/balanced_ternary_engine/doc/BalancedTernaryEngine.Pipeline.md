# `BalancedTernaryEngine.Pipeline`

End-to-end conversion: safetensors model → `.tbin` balanced ternary model.

    {:ok, summary} =
      BalancedTernaryEngine.Pipeline.run("model.safetensors", "model.tbin",
        arch: :llama,
        mode: :absmean
      )

Steps:

  1. Read all tensors from the source `.safetensors` file
     (`BalancedTernaryEngine.Safetensors`)
  2. Classify every tensor via the architecture policy
     (`BalancedTernaryEngine.Architectures`) → ternary | keep FP16 | skip
  3. Quantize ternary targets with the selected mode and pack them;
     sensitive tensors are stored as raw FP16 records
  4. Export a version-2 `.tbin` file with provenance metadata

## Options

  - `:arch` — `:llama`, `:gpt2`, `:bert`, `:vit`, `:generic`
    (default: `:generic`)
  - `:mode` — quantization mode passed to
    `BalancedTernaryEngine.Quantizer.quantize/2`: `:delta`, `:twn`,
    `:absmean` (default: `:twn`)
  - `:delta` — explicit threshold when mode is `:delta`
  - `:compress` — gzip the output (default: false)

# `run`

```elixir
@spec run(String.t(), String.t(), keyword()) :: {:ok, map()} | {:error, term()}
```

# `validate_input`

```elixir
@spec validate_input(String.t()) :: :ok | {:error, term()}
```

Validate that a conversion input path exists.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
