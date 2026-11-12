# `BalancedTernaryEngine.Architectures`

Per-architecture mixed-precision policies for common model families.

Encodes the conversion recipes from `details/13-model-conversion-guide.md`
(§13.4): which weight tensors are ternarized, which stay FP16 because
they are quantization-sensitive (embeddings, norms, LM head), and
which are skipped entirely (non-weight buffers).

Presets:

  - `:llama`   — LLaMA / Mistral / Mixtral / Qwen-style decoders
  - `:gpt2`    — GPT-2 / GPT-NeoX-style decoders (Conv1D transposition handled)
  - `:bert`    — BERT / RoBERTa encoders
  - `:vit`       — Vision Transformers
  - `:yolo`      — YOLO object detectors
  - `:imagenet`  — ImageNet image classifiers
  - `:llm`       — Decoder-only LLMs under 8B parameters
  - `:generic`   — heuristic fallback for unknown architectures

# `policy`

```elixir
@type policy() :: :ternary | :keep_fp16 | :skip
```

# `preset`

```elixir
@type preset() :: :llama | :gpt2 | :bert | :vit | :yolo | :imagenet | :llm | :generic
```

# `classify`

```elixir
@spec classify(preset() | atom(), String.t()) :: policy()
```

Classify a weight tensor by layer name.

Returns `:ternary` (convert), `:keep_fp16` (store dense at half
precision), or `:skip` (do not store).

Rules are evaluated in a fixed order — buffers first, then standalone
biases, then sensitivity, with "2-D weight" as the ternary default:

    classify(:llama, "model.layers.0.self_attn.q_proj.weight")   #=> :ternary
    classify(:llama, "model.embed_tokens.weight")                #=> :keep_fp16
    classify(:llama, "model.layers.0.mlp.down_proj.bias")        #=> :skip
    classify(:gpt2,  "transformer.wte.weight")                   #=> :keep_fp16
    classify(:bert,  "bert.pooler.dense.weight")                 #=> :keep_fp16

Matching is done on the downcased name because HuggingFace checkpoints
prefix tensors with varying model prefixes ("model.", "bert.", "vit.")
but keep suffix conventions stable.

# `presets`

```elixir
@spec presets() :: [preset()]
```

# `transform`

```elixir
@spec transform(preset() | atom(), String.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
```

Apply architecture-specific tensor transforms before quantization.

GPT-2 uses `Conv1D` modules whose weights are stored transposed
`(in_features, out_features)` relative to `nn.Linear`. They are
transposed here so every `.tbin` layer is `(out, in)`.

---

*Consult [api-reference.md](api-reference.md) for complete listing*
