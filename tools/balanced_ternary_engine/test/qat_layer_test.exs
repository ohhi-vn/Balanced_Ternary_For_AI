defmodule BalancedTernaryEngine.QATLayerTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.{Layer, QATLayer}

  test "from_pretrained creates layer with shadow weights" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    assert q.name == "fc"
    assert Nx.shape(q.weight_shadow) == {2, 2}
    assert length(Nx.to_flat_list(q.scale)) == 2
  end

  test "ternarize returns effective ternary weights" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    {eff, shadow} = QATLayer.ternarize(q)
    assert Nx.shape(eff) == {2, 2}
    vals = eff |> Nx.to_flat_list() |> Enum.map(&round/1)
    assert Enum.all?(vals, &(&1 in [-1, 0, 1]))
    assert shadow == q.weight_shadow
  end

  test "update changes shadow weights" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    grad = Nx.tensor([[1.0, 1.0], [1.0, 1.0]])
    q2 = QATLayer.update(q, grad, 0.1)
    refute q2.weight_shadow == q.weight_shadow
  end

  test "freeze produces a loadable Layer" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    layer = QATLayer.freeze(q)
    assert %Layer{} = layer
    assert layer.name == "fc"
    assert elem(layer.shape, 0) == 2
  end

  test "train_step returns updated layer and loss" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    q = QATLayer.from_pretrained(w, "fc", delta: 0.5)
    x = Nx.tensor([[1.0, 2.0]])
    target = Nx.tensor([[0.5, -0.1]])
    {q2, loss} = QATLayer.train_step(q, x, target, lr: 0.01)
    assert %QATLayer{} = q2
    assert is_float(loss)
  end
end
