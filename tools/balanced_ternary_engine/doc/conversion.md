# Conversion Guide

## Quick Path (PTQ)

```elixir
{:ok, summary} =
  BalancedTernaryEngine.Pipeline.run("model.safetensors", "model.tbin",
    arch: :llama,
    mode: :absmean
  )
```

## CLI

```bash
ternary_converter convert --input model.safetensors --arch llama --mode absmean --output model.tbin
```

## Profiles

Use `--profile yolo|imagenet|llm` for family-specific defaults.