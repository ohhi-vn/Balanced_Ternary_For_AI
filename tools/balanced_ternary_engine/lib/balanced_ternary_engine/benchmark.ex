defmodule BalancedTernaryEngine.Benchmark do
  @moduledoc """
  Simple benchmarking helpers for conversion and runtime execution.
  """

  alias BalancedTernaryEngine.{Pipeline, Runtime}

  @spec measure((-> any())) :: %{time_us: non_neg_integer(), result: any()}
  def measure(fun) when is_function(fun, 0) do
    {time, result} = :timer.tc(fun)
    %{time_us: time, result: result}
  end

  @spec benchmark_conversion(String.t(), String.t(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def benchmark_conversion(input_path, output_path, opts \\ []) do
    {time, result} = :timer.tc(fn -> Pipeline.run(input_path, output_path, opts) end)

    case result do
      {:ok, summary} -> {:ok, %{time_us: time, summary: summary}}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec benchmark_runtime(Runtime.t(), Nx.Tensor.t()) :: {:ok, map()}
  def benchmark_runtime(runtime, input) do
    {time, output} = :timer.tc(fn -> Runtime.forward(runtime, input) end)
    {:ok, %{time_us: time, output: output}}
  end
end
