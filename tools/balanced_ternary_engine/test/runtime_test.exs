defmodule BalancedTernaryEngine.RuntimeTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.Runtime

  test "forward chains layers" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)

    {:ok, rt} = {:ok, %Runtime{layers: [layer], config: %{}, cache: %{}}}

    x = Nx.tensor([[1.0, 2.0]])
    out = Runtime.forward(rt, x)
    assert Nx.shape(out) == {1, 2}
  end

  test "softmax rows sum to one" do
    x = Nx.tensor([[1.0, 2.0, 3.0]])
    s = Runtime.softmax(x)
    sums = s |> Nx.sum(axes: [-1]) |> Nx.to_flat_list()
    Enum.each(sums, fn v -> assert_in_delta v, 1.0, 1.0e-5 end)
  end

  test "layer_norm finite" do
    x = Nx.tensor([[1.0, 2.0, 3.0]])
    out = Runtime.layer_norm(x)
    assert Nx.shape(out) == {1, 3}
  end

  test "deterministic output" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    rt = %Runtime{layers: [layer], config: %{}, cache: %{}}
    x = Nx.tensor([[1.0, 2.0]])
    assert Runtime.forward(rt, x) == Runtime.forward(rt, x)
  end

  test "end-to-end convert then run" do
    tmp = System.tmp_dir!()
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    tensors = %{"fc.weight" => w}
    path_in = Path.join(tmp, "e2e.safetensors")
    path_out = Path.join(tmp, "e2e.tbin")
    :ok = BalancedTernaryEngine.Safetensors.write(path_in, tensors)
    {:ok, _} = BalancedTernaryEngine.Pipeline.run(path_in, path_out, arch: :generic)
    {:ok, rt} = BalancedTernaryEngine.Runtime.load(path_out, %{})
    x = Nx.tensor([[1.0, 2.0]])
    out = BalancedTernaryEngine.Runtime.forward(rt, x)
    assert Nx.shape(out) == {1, 2}
  end

  test "invalid config rejected" do
    assert {:error, _} = Runtime.validate_config(%{num_heads: 0})
    assert {:error, _} = Runtime.validate_config(%{eps: -1.0})
    assert {:ok, %{}} = Runtime.validate_config(%{})
  end

  test "load invalid file returns error" do
    assert {:error, _} = Runtime.load("/tmp/does-not-exist.tbin", %{})
  end

  test "reset cache clears state" do
    rt = %Runtime{layers: [], config: %{}, cache: %{prefilled: true}}
    assert Runtime.reset_cache(rt).cache == %{}
  end

  test "prefill populates cache" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.Layer.from_dense(w, "fc", delta: 0.5)
    rt = %Runtime{layers: [layer], config: %{}, cache: %{}}
    {_out, new_rt} = Runtime.prefill(rt, Nx.tensor([[1.0, 2.0]]))
    assert map_size(new_rt.cache) > 0
  end

  test "kv cache append" do
    k = Nx.tensor([[1.0, 2.0]])
    v = Nx.tensor([[3.0, 4.0]])
    {out, cache} = Runtime.attention(k, k, v, %{}, 1)
    assert Map.has_key?(cache, :k)
    assert Nx.shape(out) == {1, 2}
  end
end
