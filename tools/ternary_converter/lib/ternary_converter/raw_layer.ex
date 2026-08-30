defmodule TernaryConverter.RawLayer do
  @moduledoc """
  A dense weight matrix kept at half precision inside a converted model.

  Quantization-sensitive tensors — token embeddings, LM head, layer
  norms — are stored as `RawLayer` instead of being ternarized
  (see `TernaryConverter.Architectures`). Weights are held as raw
  float16 bytes; activations are computed in f32.
  """

  defstruct [:name, :data, :shape]

  @type t :: %__MODULE__{
          name: String.t(),
          data: binary(),
          shape: tuple()
        }

  @doc """
  Build a RawLayer from a tensor of any float type (stored as f16).
  """
  @spec from_tensor(Nx.Tensor.t(), String.t()) :: t()
  def from_tensor(tensor, name) do
    %__MODULE__{
      name: name,
      data: tensor |> Nx.as_type(:f16) |> Nx.to_binary(),
      shape: Nx.shape(tensor)
    }
  end

  @doc """
  Materialize the weights as an f32 tensor.
  """
  @spec to_tensor(t()) :: Nx.Tensor.t()
  def to_tensor(%__MODULE__{} = layer) do
    layer.data
    |> Nx.from_binary({:f, 16})
    |> Nx.as_type(:f32)
    |> Nx.reshape(layer.shape)
  end

  @doc """
  Forward pass: dequantize to f32 and apply as a linear projection.

  Handles both `(out, in)` (`y = x·Wᵀ`) and `(in, out)` (`y = x·W`)
  layouts — GPT-2-style checkpoints store Conv1D weights transposed, and
  the converter preserves whichever layout it was given. Intended for
  lm_head-style tensors; embeddings and norm vectors are consumed by
  runtimes via `to_tensor/1`.
  """
  @spec forward_nx(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward_nx(%__MODULE__{} = layer, input) do
    w = to_tensor(layer)

    if elem(layer.shape, 0) == Nx.axis_size(input, -1) do
      # Stored (out, in): contract input's last axis with W's second.
      Nx.dot(input, [tuple_size(Nx.shape(input)) - 1], Nx.transpose(w), [1])
    else
      # Stored (in, out): contract directly with W's first axis.
      Nx.dot(input, [tuple_size(Nx.shape(input)) - 1], w, [1])
    end
  end

  @doc """
  Serialize to binary: name, shape, f16 payload.
  """
  @spec to_binary(t()) :: binary()
  def to_binary(%__MODULE__{} = layer) do
    dims = Tuple.to_list(layer.shape)

    <<
      byte_size(layer.name)::16,
      layer.name::binary,
      length(dims)::8,
      for(d <- dims, into: <<>>, do: <<d::32>>)::binary,
      layer.data::binary
    >>
  end

  @doc """
  Deserialize from binary. Returns `{layer, rest_binary}`.
  """
  @spec from_binary(binary()) :: {t(), binary()}
  def from_binary(binary) do
    <<name_len::16, rest::binary>> = binary
    <<name::binary-size(^name_len), ndims::8, rest::binary>> = rest
    <<dims_bin::binary-size(^ndims * 4), data_and_rest::binary>> = rest

    dims = for <<d::32 <- dims_bin>>, do: d

    flat = Enum.product(dims)
    <<data::binary-size(^flat * 2), rest::binary>> = data_and_rest

    {
      %__MODULE__{
        name: name,
        data: data,
        shape: List.to_tuple(dims)
      },
      rest
    }
  end

  @doc """
  Byte size of the stored f16 payload (for stats).
  """
  @spec packed_bytes(t()) :: non_neg_integer()
  def packed_bytes(%__MODULE__{} = layer), do: byte_size(layer.data)
end
