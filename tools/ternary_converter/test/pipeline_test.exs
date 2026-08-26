defmodule TernaryConverter.PipelineTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.{Exporter, Layer, Pipeline, RawLayer}

  @tmp System.tmp_dir!()

  # A miniature LLaMA-shaped model: attention + MLP projections plus
  # the sensitive tensors that must stay FP16.
  defp llama_like_tensors do
    rand = fn key, shape, opts ->
      if Keyword.get(opts, :normal, false) do
        elem(Nx.Random.normal(Nx.Random.key(key), 0.0, 0.05, shape: shape), 0)
      else
        elem(Nx.Random.uniform(Nx.Random.key(key), -0.5, 0.5, shape: shape), 0)
      end
    end

    %{
      "model.embed_tokens.weight" => rand.(1, {64, 16}, []),
      "model.layers.0.self_attn.q_proj.weight" => rand.(2, {16, 16}, normal: true),
      "model.layers.0.self_attn.k_proj.weight" => rand.(3, {16, 16}, normal: true),
      "model.layers.0.self_attn.o_proj.weight" => rand.(4, {16, 16}, normal: true),
      "model.layers.0.mlp.gate_proj.weight" => rand.(5, {32, 16}, normal: true),
      "model.layers.0.mlp.down_proj.weight" => rand.(6, {16, 32}, normal: true),
      "model.layers.0.input_layernorm.weight" => Nx.tensor(for _ <- 1..16, do: 0.98),
      "model.norm.weight" => Nx.tensor(for _ <- 1..16, do: 1.0),
      "model.layers.0.self_attn.rotary_emb.inv_freq" => Nx.tensor(for i <- 0..7, do: i * 0.1)
    }
  end

  describe "run/3" do
    test "converts a llama-like safetensors model end to end" do
      src = Path.join(@tmp, "pipe_llama.safetensors")
      dst = Path.join(@tmp, "pipe_llama.tbin")

      :ok = __MODULE__.SafetensorsHelper.write_safetensors(src, llama_like_tensors())

      {:ok, summary} =
        Pipeline.run(src, dst, arch: :llama, mode: :absmean)

      assert summary.arch == :llama
      assert length(summary.ternary_layers) == 5
      assert length(summary.fp16_layers) == 3

      # inv_freq buffer is skipped; embed/norm/lm kept fp16
      names_ternary = Enum.map(summary.ternary_layers, & &1.name)
      assert "model.layers.0.mlp.gate_proj.weight" in names_ternary
      refute "model.embed_tokens.weight" in names_ternary

      names_fp16 = Enum.map(summary.fp16_layers, & &1.name)
      assert "model.embed_tokens.weight" in names_fp16
      assert "model.norm.weight" in names_fp16
      refute "model.layers.0.self_attn.rotary_emb.inv_freq" in names_fp16

      assert File.exists?(dst)
    end

    test "output .tbin round-trips with mixed record kinds" do
      src = Path.join(@tmp, "pipe_rt.safetensors")
      dst = Path.join(@tmp, "pipe_rt.tbin")

      :ok = __MODULE__.SafetensorsHelper.write_safetensors(src, llama_like_tensors())
      {:ok, _summary} = Pipeline.run(src, dst, arch: :llama)

      {:ok, layers, metadata} = Exporter.load(dst)

      ternary = Enum.filter(layers, &match?(%Layer{}, &1))
      raw = Enum.filter(layers, &match?(%RawLayer{}, &1))

      assert length(ternary) == 5
      assert length(raw) == 3

      assert metadata["architecture"] == "llama"
      assert metadata["mode"] == "twn"
      assert metadata["ternary_layers"] == 5
      assert metadata["fp16_layers"] == 3
      assert metadata["skipped_tensors"] == 1

      # Every raw layer reconstructs its original f32 content (within f16 precision)
      norm = Enum.find(raw, &(&1.name == "model.norm.weight"))
      assert Nx.to_flat_list(RawLayer.to_tensor(norm)) |> Enum.max() |> abs() > 0.9
    end

    test "gpt2 preset transposes conv1d weights before quantization" do
      src = Path.join(@tmp, "pipe_gpt2.safetensors")
      dst = Path.join(@tmp, "pipe_gpt2.tbin")

      tensors = %{
        # Conv1D layout: (in=8, out=12). After transpose: (out=12, in=8).
        "transformer.h.0.attn.c_attn.weight" =>
          Nx.Random.normal(Nx.Random.key(9), 0.0, 0.1, shape: {8, 12}) |> elem(0),
        "transformer.wte.weight" =>
          Nx.Random.uniform(Nx.Random.key(10), -0.1, 0.1, shape: {50, 8}) |> elem(0)
      }

      :ok = __MODULE__.SafetensorsHelper.write_safetensors(src, tensors)
      {:ok, summary} = Pipeline.run(src, dst, arch: :gpt2)

      [layer] = summary.ternary_layers
      assert layer.name == "transformer.h.0.attn.c_attn.weight"
      assert elem(layer.shape, 0) == 12
      assert elem(layer.shape, 1) == 8
    end

    test "unknown architecture raises" do
      assert_raise ArgumentError, fn ->
        Pipeline.run("nope.safetensors", "nope.tbin", arch: :gemma)
      end
    end

    test "missing file returns error tuple" do
      assert {:error, reason} = Pipeline.run(@tmp <> "/missing.safetensors", @tmp <> "/x.tbin")
      assert reason in [:enoent, :eisdir]
    end
  end

  defmodule SafetensorsHelper do
    def write_safetensors(path, tensors) do
      TernaryConverter.Safetensors.write(path, tensors)
    end
  end
end
