defmodule BalancedTernaryEngine.SensitivityTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.Sensitivity

  defp weights do
    w1 = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    w2 = Nx.tensor([[0.1, -0.9], [0.7, -0.4]])
    [{"a", w1}, {"b", w2}]
  end

  describe "analyze/3" do
    test "returns a list of {name, mse} sorted descending" do
      result = Sensitivity.analyze(weights(), Nx.tensor([[1.0, 2.0]]), 0.5)
      assert length(result) == 2
      assert Enum.all?(result, fn {n, m} -> is_binary(n) and is_float(m) end)
      # sorted descending by mse
      mses = Enum.map(result, fn {_, m} -> m end)
      assert mses == Enum.sort(mses, :desc)
    end
  end

  describe "auto_mixed_precision/4" do
    test "returns assignment map with counts" do
      {assignment, t, f} =
        Sensitivity.auto_mixed_precision(weights(), Nx.tensor([[1.0, 2.0]]), 0.5, 0.0)

      assert is_map(assignment)
      assert t + f == 2
      assert Map.has_key?(assignment, "a")
      assert Map.has_key?(assignment, "b")
    end

    test "high target promotes more layers" do
      {_a1, t1, _} =
        Sensitivity.auto_mixed_precision(weights(), Nx.tensor([[1.0, 2.0]]), 0.5, 0.0)

      {_a2, t2, _} =
        Sensitivity.auto_mixed_precision(weights(), Nx.tensor([[1.0, 2.0]]), 0.5, 0.99)

      assert t2 <= t1
    end
  end
end
