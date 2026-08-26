defmodule TernaryConverter.Architectures do
  @moduledoc """
  Per-architecture mixed-precision policies for common model families.

  Encodes the conversion recipes from `details/13-model-conversion-guide.md`
  (§13.4): which weight tensors are ternarized, which stay FP16 because
  they are quantization-sensitive (embeddings, norms, LM head), and
  which are skipped entirely (non-weight buffers).

  Presets:

    - `:llama`   — LLaMA / Mistral / Mixtral / Qwen-style decoders
    - `:gpt2`    — GPT-2 / GPT-NeoX-style decoders (Conv1D transposition handled)
    - `:bert`    — BERT / RoBERTa encoders
    - `:vit`     — Vision Transformers
    - `:generic` — heuristic fallback for unknown architectures
  """

  @presets [:llama, :gpt2, :bert, :vit, :generic]

  @type preset :: :llama | :gpt2 | :bert | :vit | :generic
  @type policy :: :ternary | :keep_fp16 | :skip

  @spec presets :: [preset()]
  def presets, do: @presets

  @doc """
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
  """
  @spec classify(preset() | atom(), String.t()) :: policy()
  def classify(preset, name) when preset in @presets do
    lname = String.downcase(name)

    cond do
      # Rotary tables / causal-mask helpers are runtime buffers, not
      # learned parameters — converting them would be meaningless.
      buffer?(lname) ->
        :skip

      String.contains?(name, ".bias") or not String.contains?(name, "weight") ->
        # Bias vectors are folded into inference separately; our format
        # stores them per-layer only, so standalone biases are skipped.
        :skip

      sensitive?(preset, lname) ->
        :keep_fp16

      true ->
        :ternary
    end
  end

  def classify(_preset, _name), do: raise(ArgumentError, "unknown preset")

  @doc """
  Apply architecture-specific tensor transforms before quantization.

  GPT-2 uses `Conv1D` modules whose weights are stored transposed
  `(in_features, out_features)` relative to `nn.Linear`. They are
  transposed here so every `.tbin` layer is `(out, in)`.
  """
  @spec transform(preset() | atom(), String.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def transform(:gpt2, name, tensor) do
    if Regex.match?(~r/c_(attn|proj|fc)\.weight$/, name) do
      Nx.transpose(tensor)
    else
      tensor
    end
  end

  def transform(_preset, _name, tensor), do: tensor

  # ── Sensitivity rules ────────────────────────────────────────

  # Rotary tables, causal masks and similar are buffers, not weights.
  defp buffer?(lname) do
    Regex.match?(~r/(inv_freq|rotary|masked_bias|attn\.bias|position_ids)/, lname)
  end

  defp sensitive?(:llama, lname) do
    Regex.match?(~r/(embed_tokens|embed\.|lm_head|(^|\.)norm(\.|$)|layernorm)/, lname)
  end

  defp sensitive?(:gpt2, lname) do
    Regex.match?(~r/(wte|wpe|lm_head|(^|[._])ln(_|$)|layernorm)/, lname)
  end

  defp sensitive?(:bert, lname) do
    Regex.match?(
      ~r/(embeddings|word_embed|position_embed|token_type|pooler|classifier|^cls\.|layernorm|layer_norm)/,
      lname
    )
  end

  defp sensitive?(:vit, lname) do
    Regex.match?(
      ~r/(embeddings|cls_token|pos_embed|position_embed|patch_embed|conv_proj|^norm|(^|[._])norm([._]|$)|head\.|classifier|pooler|pre_logits|layernorm|layer_norm)/,
      lname
    )
  end

  defp sensitive?(:generic, lname) do
    Regex.match?(
      ~r/(embed|(^|[._])norm([._]|$)|layernorm|(^|[._])ln([._]|$)|lm_head|\bhead\b|classifier|cls)/,
      lname
    )
  end
end
