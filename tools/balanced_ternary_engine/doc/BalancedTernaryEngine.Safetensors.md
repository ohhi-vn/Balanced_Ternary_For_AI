# `BalancedTernaryEngine.Safetensors`

Pure-Elixir reader/writer for the HuggingFace `.safetensors` format.

Layout:

    [header_len: u64 little-endian][header: UTF-8 JSON][tensor data]

Header JSON maps tensor names to `%{"dtype" => ..., "shape" => [...],
"data_offsets" => [begin, end]}` where offsets are relative to the
start of the data section. An optional `__metadata__` key holds a
string-to-string map.

Float tensors are returned as `:f32`; integer tensors as `:s64`.

# `decode`

```elixir
@spec decode(binary()) :: {:ok, map(), map()} | {:error, term()}
```

Decode safetensors from a binary.

Any structural problem (truncated file, bad JSON, bad offsets) returns
`{:error, :invalid_safetensors}`; an unknown dtype raises `ArgumentError`
since it indicates a caller/producer bug rather than corrupt input.

# `encode`

```elixir
@spec encode(%{required(String.t()) =&gt; Nx.Tensor.t()}, %{
  required(String.t()) =&gt; String.t()
}) ::
  binary()
```

Encode a map of tensors to safetensors binary.

Supported tensor types are written as-is; anything else is converted
to f32. Data blocks are padded to 8-byte alignment.

# `read`

```elixir
@spec read(String.t() | binary()) ::
  {:ok, %{required(String.t()) =&gt; Nx.Tensor.t()},
   %{required(String.t()) =&gt; String.t()}}
  | {:error, term()}
```

Read a safetensors file.

Returns `{:ok, tensors, metadata}` where `tensors` is
`%{name => Nx.Tensor.t()}` (floats converted to f32) and `metadata`
is the optional `__metadata__` map (empty map if absent).

# `read_sharded`

Read a sharded safetensors model given a path to its index JSON file
(`model.safetensors.index.json`) or a directory containing one.

Resolves shard paths relative to the index file's directory, loads every
shard, and merges their tensors into a single map.

# `write`

```elixir
@spec write(String.t(), %{required(String.t()) =&gt; Nx.Tensor.t()}, %{
  required(String.t()) =&gt; String.t()
}) ::
  :ok | {:error, term()}
```

Write a map of tensors to a `.safetensors` file.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
