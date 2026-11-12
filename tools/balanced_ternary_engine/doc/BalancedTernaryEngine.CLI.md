# `BalancedTernaryEngine.CLI`

Command-line interface for the BalancedTernaryEngine tool.

Four commands share one option parser:

  - `convert` — safetensors checkpoint → `.tbin` (or synthetic model
    when no `--input` is given)
  - `demo` — quantize/pack/GEMM walkthrough on random weights
  - `info` — inspect a `.tbin` file
  - `validate` — round-trip + per-layer forward verification

Options are parsed with `OptionParser` under `strict:` mode, so typos
like `--deltta` are reported instead of being silently ignored.

# `main`

```elixir
@spec main(OptionParser.parsed() | [String.t()]) :: :ok
```

---

*Consult [api-reference.md](api-reference.md) for complete listing*
