# `BalancedTernaryEngine.Exporter`

Exports and imports ternary models to/from .tbin binary format.

The .tbin format:

    [Header][Record 1][Record 2]...[Record N]

    Header:
      magic:       4 bytes   ("TBN\0")
      version:     4 bytes   (uint32; 2 as of mixed-precision support)
      num_layers:  4 bytes   (uint32)
      meta_len:    4 bytes   (uint32)
      metadata:    meta_len bytes (UTF-8 JSON)

    Record (version >= 2):
      kind:        1 byte    (0 = ternary, 1 = raw FP16)
      size:        4 bytes   (uint32)
      payload:     size bytes
        - ternary: BalancedTernaryEngine.Layer.to_binary/1
        - raw:     BalancedTernaryEngine.RawLayer.to_binary/1

    Record (version 1):
      size:        4 bytes   (uint32)
      payload:     size bytes (ternary layer only)

Version 1 files remain readable; they are written as version 2 on the
next export.

# `export`

```elixir
@spec export([struct()], String.t(), keyword()) :: :ok | {:error, term()}
```

Export a list of layers to a .tbin file.

Layers may be `BalancedTernaryEngine.Layer` (ternary) or
`BalancedTernaryEngine.RawLayer` (kept at FP16); each is tagged with its
record kind.

## Options

  - `:metadata` — Map of metadata (model name, version, etc.)
  - `:compress` — Whether to compress with gzip (default: false)

# `format_version`

```elixir
@spec format_version(String.t()) :: {:ok, non_neg_integer()} | {:error, term()}
```

Return the format version stored in a `.tbin` file.

# `load`

```elixir
@spec load(String.t()) :: {:ok, [struct()], map()} | {:error, term()}
```

Load a .tbin file back into a list of layers.

# `validate`

```elixir
@spec validate(String.t()) :: :ok | {:error, term()}
```

Validate that a `.tbin` file can be loaded.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
