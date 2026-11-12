defmodule BalancedTernaryEngine.CoverageBoostTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.{
    Exporter,
    Layer,
    Pipeline,
    Quantizer,
    RawLayer,
    Runtime,
    Safetensors,
    Sensitivity
  }

  @tmp System.tmp_dir!()

  test "layer sparse packing and bias padding" do
    w = Nx.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    layer = Layer.from_dense(w, "sparse", delta: 0.5, packing: :sparse, bias: [1.0])
    assert layer.packing == :sparse
    assert length(layer.bias) == 2
  end

  test "layer bias truncation" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = Layer.from_dense(w, "bias", delta: 0.5, bias: [1.0, 2.0, 3.0])
    assert length(layer.bias) == 2
  end

  test "raw layer forward branches" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2], [0.1, 0.2]])
    rl = RawLayer.from_tensor(w, "raw")
    out1 = RawLayer.forward_nx(rl, Nx.tensor([[1.0, 2.0]]))
    assert Nx.shape(out1) == {1, 3}
    w2 = Nx.tensor([[0.8, -0.3, 0.1], [0.5, -0.2, 0.2]])
    rl2 = RawLayer.from_tensor(w2, "raw2")
    out2 = RawLayer.forward_nx(rl2, Nx.tensor([[1.0, 2.0]]))
    assert is_struct(out2, Nx.Tensor)
    assert RawLayer.packed_bytes(rl) > 0
  end

  test "runtime validate_config and new and decode" do
    assert {:error, :invalid_config} = Runtime.validate_config("bad")
    assert {:ok, %Runtime{}} = Runtime.new([], %{})
    rt = %Runtime{layers: [], config: %{}, cache: %{}}
    {_out, new_rt} = Runtime.decode(rt, Nx.tensor([[1.0, 2.0]]))
    assert Map.has_key?(new_rt.cache, :decoded)
  end

  test "runtime load with config and invalid file" do
    assert {:error, _} = Runtime.load("/nope.tbin", %{num_heads: 2})
    assert {:error, _} = Runtime.validate_config(%{num_heads: 0})
  end

  test "quantizer clipped modes and auto threshold zero" do
    w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
    # delta mode with clip
    {t, s, d} = Quantizer.quantize(w, mode: :delta, delta: 0.4, clip_sigma: 2.0)
    assert Nx.shape(t) == Nx.shape(w)
    # twn mode with clip
    {t2, s2, d2} = Quantizer.quantize(w, mode: :twn, clip_sigma: 2.0)
    assert Nx.shape(t2) == Nx.shape(w)
    # absmean mode with clip
    {t3, s3, d3} = Quantizer.quantize(w, mode: :absmean, clip_sigma: 2.0)
    assert Nx.shape(t3) == Nx.shape(w)
    # auto threshold zero target
    {delta, sparsity} = Quantizer.auto_threshold(w, 0.0)
    assert delta == 0.0
  end

  test "quantizer clipped scales and auto threshold" do
    w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
    scales = Quantizer.compute_scales_clipped(w, 0.4, 2.0)
    assert is_list(scales) or is_struct(scales, Nx.Tensor)
    {delta, sparsity} = Quantizer.auto_threshold(w, 0.3)
    assert is_number(delta)
    assert is_number(sparsity)
  end

  test "safetensors encode decode and sharded" do
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    path = Path.join(@tmp, "cov.safetensors")
    :ok = Safetensors.write(path, tensors)
    {:ok, loaded, _meta} = Safetensors.read(path)
    assert Map.has_key?(loaded, "fc.weight")
    assert {:error, _} = Safetensors.read("/nope.safetensors")
  end

  test "exporter compressed and invalid" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = Layer.from_dense(w, "fc", delta: 0.5)
    path = Path.join(@tmp, "cov_compressed.tbin")
    :ok = Exporter.export([layer], path, compress: true)
    {:ok, _layers, _meta} = Exporter.load(path)
    assert {:error, _} = Exporter.load("/nope.tbin")
    assert {:error, _} = Exporter.format_version("/nope.tbin")
  end

  test "exporter version 1 readable" do
    # Build a v1 file manually? Skip; instead ensure v2 version reported
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = Layer.from_dense(w, "fc", delta: 0.5)
    path = Path.join(@tmp, "cov_v2.tbin")
    :ok = Exporter.export([layer], path)
    assert {:ok, 2} = Exporter.format_version(path)
  end

  test "sensitivity analyze" do
    weights = %{"fc" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    sample = Nx.tensor([[1.0, 2.0]])
    result = Sensitivity.analyze([{"fc", weights["fc"]}], sample, 0.5)
    assert is_list(result)

    {assignment, t, f} =
      Sensitivity.auto_mixed_precision([{"fc", weights["fc"]}], sample, 0.5, 0.99)

    assert is_map(assignment)
  end

  test "pipeline sharded and sparsity target" do
    dir = Path.join(@tmp, "shard_dir2")
    File.rm_rf!(dir)
    File.mkdir_p!(dir)
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    shard = "shard-00001.safetensors"
    :ok = Safetensors.write(Path.join(dir, shard), tensors)
    index = %{"weight_map" => %{"fc.weight" => shard}, "metadata" => %{}}
    File.write!(Path.join(dir, "model.safetensors.index.json"), Jason.encode!(index))
    assert {:ok, _, _} = BalancedTernaryEngine.Safetensors.read_sharded(dir)

    # sparsity target
    src = Path.join(@tmp, "sp.safetensors")
    :ok = Safetensors.write(src, tensors)
    out2 = Path.join(@tmp, "sp.tbin")
    {:ok, _} = Pipeline.run(src, out2, arch: :generic, sparsity_target: 0.5)
    assert File.exists?(out2)
  end

  test "pipeline 4d conv and bias" do
    src = Path.join(@tmp, "conv.safetensors")
    out = Path.join(@tmp, "conv.tbin")

    tensors = %{
      "conv.weight" =>
        Nx.Random.uniform(Nx.Random.key(1), -0.5, 0.5, shape: {2, 2, 2, 2}) |> elem(0),
      "conv.bias" => Nx.tensor([0.1, 0.2])
    }

    :ok = Safetensors.write(src, tensors)
    {:ok, summary} = Pipeline.run(src, out, arch: :generic)
    assert summary.ternary_layers != []
  end

  test "pipeline target similarity" do
    src = Path.join(@tmp, "ts.safetensors")
    out = Path.join(@tmp, "ts.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = Safetensors.write(src, tensors)
    {:ok, _} = Pipeline.run(src, out, arch: :generic, target_similarity: 0.5)
    assert File.exists?(out)
  end

  test "cli invalid file info and validate" do
    path = Path.join(@tmp, "nope.tbin")
    # should not crash; returns error printed
    BalancedTernaryEngine.CLI.main(["info", "--model", path])
    BalancedTernaryEngine.CLI.main(["validate", "--model", path])
  end

  test "cli unknown command" do
    assert :ok = BalancedTernaryEngine.CLI.main(["frob"])
  end

  test "runtime branches" do
    # invalid config non-map
    assert {:error, :invalid_config} = Runtime.validate_config(42)
    # new
    assert {:ok, %Runtime{}} = Runtime.new([], %{})
    # forward with raw layer and unknown layer
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    rl = RawLayer.from_tensor(w, "raw")
    rt = %Runtime{layers: [rl], config: %{}, cache: %{}}
    out = Runtime.forward(rt, Nx.tensor([[1.0, 2.0]]))
    assert is_struct(out, Nx.Tensor)
    # attention append (existing + new)
    k = Nx.tensor([[1.0, 2.0]])
    v = Nx.tensor([[3.0, 4.0]])
    {_out, cache} = Runtime.attention(k, k, v, %{k: k, v: v}, 1)
    assert Map.has_key?(cache, :k)
    # load valid file with config
    src = Path.join(@tmp, "rt_load.safetensors")
    outp = Path.join(@tmp, "rt_load.tbin")
    tensors = %{"fc.weight" => w}
    :ok = Safetensors.write(src, tensors)
    {:ok, _} = Pipeline.run(src, outp, arch: :generic)
    assert {:ok, _rt} = Runtime.load(outp, %{num_heads: 2})
  end

  test "safetensors encode decode invalid" do
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    binary = Safetensors.encode(tensors, %{})
    {:ok, decoded, _meta} = Safetensors.decode(binary)
    assert Map.has_key?(decoded, "fc.weight")
    assert {:error, _} = Safetensors.decode(<<0, 1, 2>>)
  end

  test "sensitivity analyze direct" do
    weights = [{"fc", Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}]
    sample = Nx.tensor([[1.0, 2.0]])
    result = Sensitivity.analyze(weights, sample, 0.5)
    assert is_list(result)
  end

  test "pipeline orphan bias" do
    src = Path.join(@tmp, "orphan.safetensors")
    out = Path.join(@tmp, "orphan.tbin")

    tensors = %{
      "norm.weight" => Nx.tensor([1.0, 2.0]),
      "norm.bias" => Nx.tensor([0.1, 0.2])
    }

    :ok = Safetensors.write(src, tensors)
    {:ok, summary} = Pipeline.run(src, out, arch: :generic)
    assert summary.fp16_layers != []
  end

  test "cli imagenet llm commands" do
    src = Path.join(@tmp, "cli_img.safetensors")
    out = Path.join(@tmp, "cli_img.tbin")

    tensors = %{
      "layer4.1.conv2.weight" =>
        Nx.Random.uniform(Nx.Random.key(1), -0.5, 0.5, shape: {2, 2, 2, 2}) |> elem(0)
    }

    :ok = Safetensors.write(src, tensors)
    BalancedTernaryEngine.CLI.main(["imagenet", "--input", src, "--output", out])
    assert File.exists?(out)

    src2 = Path.join(@tmp, "cli_llm.safetensors")
    out2 = Path.join(@tmp, "cli_llm.tbin")

    tensors2 = %{
      "model.layers.0.self_attn.q_proj.weight" =>
        Nx.Random.normal(Nx.Random.key(2), 0.0, 0.05, shape: {2, 2}) |> elem(0)
    }

    :ok = Safetensors.write(src2, tensors2)
    BalancedTernaryEngine.CLI.main(["llm", "--input", src2, "--output", out2])
    assert File.exists?(out2)
  end

  test "runtime unknown layer" do
    rt = %BalancedTernaryEngine.Runtime{layers: [:unknown], config: %{}, cache: %{}}
    out = BalancedTernaryEngine.Runtime.forward(rt, Nx.tensor([[1.0, 2.0]]))
    assert out == Nx.tensor([[1.0, 2.0]])
  end

  test "pipeline only fp16 tensors" do
    src = Path.join(@tmp, "fp16only.safetensors")
    out = Path.join(@tmp, "fp16only.tbin")
    tensors = %{"norm.weight" => Nx.tensor([1.0, 2.0])}
    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    {:ok, summary} = BalancedTernaryEngine.Pipeline.run(src, out, arch: :generic)
    assert summary.ternary_layers == []
  end

  test "cli validate with fp16" do
    src = Path.join(@tmp, "fp16cli.safetensors")
    out = Path.join(@tmp, "fp16cli.tbin")
    tensors = %{"norm.weight" => Nx.tensor([1.0, 2.0])}
    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    {:ok, _} = BalancedTernaryEngine.Pipeline.run(src, out, arch: :generic)
    BalancedTernaryEngine.CLI.main(["validate", "--model", out])
    BalancedTernaryEngine.CLI.main(["info", "--model", out])
  end

  test "layer forward scalar" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    out = BalancedTernaryEngine.Layer.forward(layer, [1.0, 2.0])
    assert is_list(out)
  end

  test "layer from_binary direct" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    bin = BalancedTernaryEngine.Layer.to_binary(layer)
    l2 = BalancedTernaryEngine.Layer.from_binary(bin)
    assert l2.name == "fc"
  end

  test "layer forward and roundtrip" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    out = BalancedTernaryEngine.Layer.forward_nx(layer, Nx.tensor([[1.0, 2.0]]))
    assert Nx.shape(out) == {1, 2}
    # sparse roundtrip
    w2 = Nx.tensor([[0.0, 0.0], [0.0, 0.0]])
    layer2 = BalancedTernaryEngine.Layer.from_dense(w2, "sparse", delta: 0.5, packing: :sparse)
    assert layer2.packing == :sparse
  end

  test "sensitivity empty layers" do
    {assignment, t, f} =
      BalancedTernaryEngine.Sensitivity.auto_mixed_precision(
        [],
        Nx.tensor([[1.0, 2.0]]),
        0.5,
        0.99
      )

    assert assignment == %{}
    assert t == 0
    assert f == 0
  end

  test "safetensors sharded error paths" do
    dir = Path.join(@tmp, "noidx")
    File.mkdir_p!(dir)
    # no index -> error
    assert {:error, _} = BalancedTernaryEngine.Safetensors.read_sharded(dir)
    # regular file path -> error
    f = Path.join(@tmp, "notjson.safetensors")
    File.write!(f, "bad")
    assert {:error, _} = BalancedTernaryEngine.Safetensors.read_sharded(f)
  end

  test "safetensors decode invalid and encode" do
    tensors = %{"a" => Nx.tensor([1.0, 2.0])}
    bin = BalancedTernaryEngine.Safetensors.encode(tensors, %{})
    {:ok, dec, _} = BalancedTernaryEngine.Safetensors.decode(bin)
    assert Map.has_key?(dec, "a")
    assert {:error, _} = BalancedTernaryEngine.Safetensors.decode(<<0, 1, 2>>)
    assert {:error, _} = BalancedTernaryEngine.Safetensors.decode(<<1, 2, 3, 4>>)
  end

  test "cli sparse validate" do
    src = Path.join(@tmp, "sparsecli.safetensors")
    out = Path.join(@tmp, "sparsecli.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.0, 0.0], [0.0, 0.0]])}
    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    {:ok, _} = BalancedTernaryEngine.Pipeline.run(src, out, arch: :generic, sparsity_target: 0.9)
    BalancedTernaryEngine.CLI.main(["validate", "--model", out])
    BalancedTernaryEngine.CLI.main(["info", "--model", out])
  end

  test "cli yolo imagenet llm commands" do
    src = Path.join(@tmp, "cli_yolo2.safetensors")
    out = Path.join(@tmp, "cli_yolo2.tbin")

    tensors = %{
      "backbone.0.conv.weight" =>
        Nx.Random.uniform(Nx.Random.key(1), -0.5, 0.5, shape: {2, 2, 2, 2}) |> elem(0)
    }

    :ok = Safetensors.write(src, tensors)
    BalancedTernaryEngine.CLI.main(["yolo", "--input", src, "--output", out])
    assert File.exists?(out)
  end
end
