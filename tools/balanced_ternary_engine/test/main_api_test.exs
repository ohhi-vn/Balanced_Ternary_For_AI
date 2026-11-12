defmodule BalancedTernaryEngine.MainApiTest do
  use ExUnit.Case, async: true

  test "convert_layer and convert_all" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.convert_layer(w, "fc", delta: 0.5)
    assert %BalancedTernaryEngine.Layer{} = layer

    layers = BalancedTernaryEngine.convert_all(%{"a" => w}, delta: 0.5)
    assert length(layers) == 1
  end

  test "inference" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.convert_layer(w, "fc", delta: 0.5)
    out = BalancedTernaryEngine.inference([layer], Nx.tensor([[1.0, 2.0]]))
    assert Nx.shape(out) == {1, 2}
  end

  test "analyze and auto mixed precision" do
    weights = %{"fc" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    sample = Nx.tensor([[1.0, 2.0]])
    sens = BalancedTernaryEngine.analyze_sensitivity(weights, sample, 0.5)
    assert is_list(sens)

    {assignment, _t, _f} = BalancedTernaryEngine.auto_mixed_precision(weights, sample, 0.5, 0.99)
    assert is_map(assignment)
  end

  test "export load stats validate" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = BalancedTernaryEngine.convert_layer(w, "fc", delta: 0.5)
    path = Path.join(System.tmp_dir!(), "main_api.tbin")
    :ok = BalancedTernaryEngine.export([layer], path)
    {:ok, loaded, _meta} = BalancedTernaryEngine.load(path)
    assert length(loaded) == 1
    stats = BalancedTernaryEngine.stats(loaded)
    assert is_map(stats)
    result = BalancedTernaryEngine.validate(loaded, %{"fc" => w}, Nx.tensor([[1.0, 2.0]]))
    assert is_map(result)
  end
end
