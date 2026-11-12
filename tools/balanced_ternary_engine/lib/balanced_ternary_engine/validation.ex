defmodule BalancedTernaryEngine.Validation do
  @moduledoc """
  Reusable validation helpers for tensors, configs, and model files.
  """

  alias BalancedTernaryEngine.{Exporter, Runtime}

  @spec validate_tensor(term()) :: {:ok, Nx.Tensor.t()} | {:error, term()}
  def validate_tensor(%Nx.Tensor{} = tensor) do
    if Nx.size(tensor) > 0 do
      {:ok, tensor}
    else
      {:error, :empty_tensor}
    end
  end

  def validate_tensor(_), do: {:error, :invalid_tensor}

  @spec validate_config(term()) :: {:ok, map()} | {:error, term()}
  def validate_config(config), do: Runtime.validate_config(config)

  @spec validate_model_file(String.t()) :: :ok | {:error, term()}
  def validate_model_file(path), do: Exporter.validate(path)
end
