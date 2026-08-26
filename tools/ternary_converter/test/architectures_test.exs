defmodule TernaryConverter.ArchitecturesTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.Architectures

  describe "classify/2 for :llama" do
    test "attention and MLP projections are ternary" do
      for name <- [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
            "model.layers.5.self_attn.v_proj.weight",
            "model.layers.9.self_attn.o_proj.weight",
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
            "model.layers.3.mlp.down_proj.weight"
          ] do
        assert Architectures.classify(:llama, name) == :ternary
      end
    end

    test "embeddings, norms, and lm_head stay FP16" do
      for name <- [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight"
          ] do
        assert Architectures.classify(:llama, name) == :keep_fp16
      end
    end

    test "rotary buffers and biases are skipped" do
      assert Architectures.classify(:llama, "model.layers.0.self_attn.rotary_emb.inv_freq") ==
               :skip

      assert Architectures.classify(:llama, "model.layers.0.mlp.down_proj.bias") == :skip
    end
  end

  describe "classify/2 for :gpt2" do
    test "conv1d weights are ternary, embeddings/norms FP16" do
      assert Architectures.classify(:gpt2, "transformer.h.0.attn.c_attn.weight") == :ternary
      assert Architectures.classify(:gpt2, "transformer.h.0.attn.c_proj.weight") == :ternary
      assert Architectures.classify(:gpt2, "transformer.h.0.mlp.c_fc.weight") == :ternary

      assert Architectures.classify(:gpt2, "transformer.wte.weight") == :keep_fp16
      assert Architectures.classify(:gpt2, "transformer.wpe.weight") == :keep_fp16
      assert Architectures.classify(:gpt2, "transformer.ln_f.weight") == :keep_fp16
      assert Architectures.classify(:gpt2, "transformer.h.0.ln_1.weight") == :keep_fp16
      assert Architectures.classify(:gpt2, "lm_head.weight") == :keep_fp16
    end

    test "transform/3 transposes conv1d weights only" do
      w = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

      assert Nx.shape(Architectures.transform(:gpt2, "h.0.attn.c_attn.weight", w)) == {2, 2}
      assert Nx.to_flat_list(Architectures.transform(:gpt2, "h.0.attn.c_attn.weight", w)) ==
               [1.0, 3.0, 2.0, 4.0]

      # lm_head is nn.Linear — untouched
      assert Architectures.transform(:gpt2, "lm_head.weight", w) == w

      # Non-gpt2 presets never transpose
      assert Architectures.transform(:llama, "h.0.attn.c_attn.weight", w) == w
    end
  end

  describe "classify/2 for :bert" do
    test "dense layers ternary; embeddings/pooler/LayerNorm FP16" do
      assert Architectures.classify(:bert, "bert.encoder.layer.0.attention.self.query.weight") ==
               :ternary

      assert Architectures.classify(:bert, "bert.encoder.layer.0.attention.output.dense.weight") ==
               :ternary

      assert Architectures.classify(:bert, "bert.encoder.layer.0.intermediate.dense.weight") ==
               :ternary

      assert Architectures.classify(:bert, "bert.encoder.layer.0.output.dense.weight") ==
               :ternary

      assert Architectures.classify(:bert, "bert.embeddings.word_embeddings.weight") ==
               :keep_fp16

      assert Architectures.classify(:bert, "bert.pooler.dense.weight") == :keep_fp16
      assert Architectures.classify(:bert, "bert.encoder.layer.0.output.LayerNorm.weight") ==
               :keep_fp16
    end
  end

  describe "classify/2 for :vit" do
    test "qkv/mlp ternary; patch embed/norms/head FP16" do
      assert Architectures.classify(:vit, "vit.encoder.layer.0.attention.attention.key.weight") ==
               :ternary

      assert Architectures.classify(:vit, "vit.encoder.layer.5.mlp.fc1.weight") == :ternary

      assert Architectures.classify(:vit, "vit.embeddings.patch_embeddings.projection.weight") ==
               :keep_fp16

      assert Architectures.classify(:vit, "vit.layernorm.weight") == :keep_fp16
      assert Architectures.classify(:vit, "classifier.weight") == :keep_fp16
    end
  end

  describe "classify/2 for :generic" do
    test "plain 2D weights are ternary, sensitive names kept FP16" do
      assert Architectures.classify(:generic, "fc1.weight") == :ternary
      assert Architectures.classify(:generic, "encoder.embedding.weight") == :keep_fp16
      assert Architectures.classify(:generic, "final_norm.weight") == :keep_fp16
      assert Architectures.classify(:generic, "some.bias") == :skip
    end

    test "unknown preset raises" do
      assert_raise ArgumentError, fn -> Architectures.classify(:gemma, "x.weight") end
    end
  end
end
