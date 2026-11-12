defmodule BalancedTernaryEngine.ProductionTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.{Benchmark, Exporter, Pipeline, QATLayer, Runtime, Validation}

  @tmp System.tmp_dir!()

  test "validation helpers" do
    assert {:ok, _} = Validation.validate_tensor(Nx.tensor([1.0]))
    assert {:error, _} = Validation.validate_tensor(42)
    assert {:error, _} = Validation.validate_tensor("bad")
    assert {:ok, %{}} = Validation.validate_config(%{})
    assert {:error, _} = Validation.validate_config(%{num_heads: 0})
  end

  test "exporter validate" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    path = Path.join(@tmp, "prod.tbin")
    :ok = BalancedTernaryEngine.Exporter.export([layer], path)
    assert :ok = Exporter.validate(path)
    assert {:error, _} = Exporter.validate("/nope.tbin")
  end

  test "model file validation" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    path = Path.join(@tmp, "prod2.tbin")
    :ok = BalancedTernaryEngine.Exporter.export([layer], path)
    assert :ok = Validation.validate_model_file(path)
    assert {:error, _} = Validation.validate_model_file("/nope.tbin")
  end

  test "benchmark helpers" do
    res = Benchmark.measure(fn -> 1 + 1 end)
    assert Map.has_key?(res, :time_us)
    assert res.result == 2

    src = Path.join(@tmp, "bench.safetensors")
    out = Path.join(@tmp, "bench.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    assert {:ok, _} = Benchmark.benchmark_conversion(src, out, arch: :generic)

    assert {:error, _} =
             Benchmark.benchmark_conversion("/nope.safetensors", Path.join(@tmp, "x.tbin"))

    assert File.exists?(out)

    {:ok, rt} = Runtime.load(out, %{})
    assert {:ok, _} = Benchmark.benchmark_runtime(rt, Nx.tensor([[1.0, 2.0]]))
  end

  test "qat validate frozen" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    assert {:ok, _layer} = QATLayer.validate_frozen(q)
  end

  test "runtime validate" do
    assert :ok = Runtime.validate(%Runtime{layers: [], config: %{}, cache: %{}})
    assert {:error, _} = Runtime.validate("bad")
  end

  test "pipeline validate input" do
    assert :ok = Pipeline.validate_input(__ENV__.file)
    assert {:error, _} = Pipeline.validate_input("/nope.safetensors")
  end

  test "malformed conversion returns error" do
    assert {:error, _} = Pipeline.run("/nope.safetensors", Path.join(@tmp, "x.tbin"))
  end
end
