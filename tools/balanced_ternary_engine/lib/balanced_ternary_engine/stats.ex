defmodule BalancedTernaryEngine.Stats do
  @moduledoc """
  Aggregate statistics over a list of model layers.

  Accepts mixed lists of `BalancedTernaryEngine.Layer` (ternary) and
  `BalancedTernaryEngine.RawLayer` (FP16) records. Size/sparsity metrics are
  computed over the ternary subset; FP16 records contribute their byte
  size to the packed total.
  """

  alias BalancedTernaryEngine.{Layer, RawLayer}

  @spec compute([struct()]) :: %{
          num_layers: non_neg_integer(),
          ternary_layers: non_neg_integer(),
          fp16_layers: non_neg_integer(),
          total_parameters: non_neg_integer(),
          nonzero_parameters: non_neg_integer(),
          zero_parameters: non_neg_integer(),
          overall_sparsity: float(),
          original_size_mb: float(),
          packed_size_mb: float(),
          compression_ratio: float()
        }
  def compute(layers) do
    ternary = Enum.filter(layers, &match?(%Layer{}, &1))
    fp16 = Enum.filter(layers, &match?(%RawLayer{}, &1))

    total_params = Enum.sum(Enum.map(ternary, &Tuple.product(&1.shape)))

    total_zeros =
      Enum.sum(Enum.map(ternary, fn l -> round(Tuple.product(l.shape) * l.sparsity) end))

    total_nonzero = total_params - total_zeros

    original_bytes = total_params * 4

    ternary_packed_bytes =
      Enum.reduce(ternary, 0, fn l, acc ->
        acc + byte_size(l.weight_packed) + length(l.scales) * 2
      end)

    fp16_bytes =
      Enum.reduce(fp16, 0, fn l, acc -> acc + RawLayer.packed_bytes(l) end)

    packed_total = ternary_packed_bytes + fp16_bytes

    %{
      num_layers: length(layers),
      ternary_layers: length(ternary),
      fp16_layers: length(fp16),
      total_parameters: total_params,
      nonzero_parameters: total_nonzero,
      zero_parameters: total_zeros,
      overall_sparsity: total_zeros / max(total_params, 1),
      original_size_mb: original_bytes / 1024 / 1024,
      packed_size_mb: packed_total / 1024 / 1024,
      compression_ratio: original_bytes / max(packed_total, 1)
    }
  end
end
