defmodule BalancedTernaryEngine.QATLayer do
  @moduledoc """
  Quantization-Aware Training layer with ternary weights.

  Maintains FP32 shadow weights that are ternarized on each forward pass.
  Uses a straight-through estimator (STE) so gradients flow to the shadow
  weights during training: the forward pass uses ternary weights, while
  the gradient is applied to the shadow weights directly.

  This module provides the primitives (ternarize, forward, update, freeze).
  The training loop composes these with a loss and an optimizer step.
  """

  defstruct [
    :name,
    :weight_shadow,
    :scale,
    :bias,
    :delta
  ]

  @type t :: %__MODULE__{
          name: String.t(),
          weight_shadow: Nx.Tensor.t(),
          scale: Nx.Tensor.t(),
          bias: Nx.Tensor.t(),
          delta: float()
        }

  alias BalancedTernaryEngine.{Layer, Quantizer}

  @doc """
  Create a QAT layer from a pretrained weight tensor.
  """
  @spec from_pretrained(Nx.Tensor.t(), String.t(), keyword()) :: t()
  def from_pretrained(weight_tensor, name, opts \\ []) do
    delta = Keyword.get(opts, :delta, 0.5)
    {out_features, _in_features} = Nx.shape(weight_tensor)

    scales = Quantizer.compute_scales(weight_tensor, delta)
    bias = Nx.broadcast(0.0, {out_features})

    %__MODULE__{
      name: name,
      weight_shadow: weight_tensor |> Nx.as_type(:f32),
      scale: scales |> Nx.as_type(:f32),
      bias: bias |> Nx.as_type(:f32),
      delta: delta
    }
  end

  @doc """
  Ternarize shadow weights. Returns `{effective, shadow}` where `effective`
  is the ternary (scaled) weight used in the forward pass and `shadow` is
  the FP32 shadow weight retained for gradient updates (STE).
  """
  @spec ternarize(t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def ternarize(%__MODULE__{} = layer) do
    w_norm = Nx.divide(layer.weight_shadow, Nx.new_axis(layer.scale, 1))
    t = Quantizer.ternarize(w_norm, layer.delta)
    w_effective = Nx.multiply(t, Nx.new_axis(layer.scale, 1))
    {w_effective, layer.weight_shadow}
  end

  @doc """
  Forward pass: output = (ternary_weight @ input) + bias.
  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{} = layer, input) do
    {w_effective, _shadow} = ternarize(layer)

    Nx.dot(input, Nx.transpose(w_effective))
    |> Nx.add(Nx.new_axis(layer.bias, 0))
  end

  @doc """
  SGD update of shadow weights: w_shadow = w_shadow - lr * grad.
  """
  @spec update(t(), Nx.Tensor.t(), float()) :: t()
  def update(%__MODULE__{} = layer, grad, lr) do
    new_shadow = Nx.subtract(layer.weight_shadow, Nx.multiply(grad, lr))
    %{layer | weight_shadow: new_shadow}
  end

  @doc """
  Apply a straight-through training step using a finite-difference
  gradient of the supplied loss function.

  `loss_fn` receives the layer and returns a scalar loss. The gradient
  w.r.t. the shadow weights is approximated and used to update them.
  Returns the updated layer and the loss value.
  """
  @spec train_step(t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          {t(), float()}
  def train_step(layer, input, target, opts \\ []) do
    lr = Keyword.get(opts, :lr, 0.01)
    loss_fn = Keyword.get(opts, :loss_fn, fn l -> default_loss(l, input, target) end)

    loss0 = loss_fn.(layer)
    eps = 1.0e-3
    shadow = layer.weight_shadow
    shape = Nx.shape(shadow)
    flat_size = Nx.size(shadow) |> Nx.to_number()

    grad_flat =
      Enum.map(0..(flat_size - 1), fn i ->
        plus = perturb(shadow, i, eps)
        minus = perturb(shadow, i, -eps)
        l_plus = loss_fn.(%{layer | weight_shadow: plus})
        l_minus = loss_fn.(%{layer | weight_shadow: minus})
        (l_plus - l_minus) / (2 * eps)
      end)

    grad = grad_flat |> Nx.tensor() |> Nx.reshape(shape) |> Nx.as_type(:f32)
    updated = update(layer, grad, lr)
    {updated, loss0}
  end

  defp default_loss(layer, input, target) do
    out = forward(layer, input)
    diff = Nx.subtract(out, target)
    Nx.mean(Nx.pow(diff, 2)) |> Nx.to_number()
  end

  defp perturb(tensor, idx, eps) do
    flat = Nx.to_flat_list(tensor)
    new_flat = List.replace_at(flat, idx, Enum.at(flat, idx) + eps)
    new_flat |> Nx.tensor() |> Nx.reshape(Nx.shape(tensor)) |> Nx.as_type(:f32)
  end

  @doc """
  Freeze a QAT layer into a standard BalancedTernaryEngine.Layer for inference.
  """
  @spec freeze(t()) :: Layer.t()
  def freeze(%__MODULE__{} = qat_layer) do
    {_w_effective, _shadow} = ternarize(qat_layer)

    Layer.from_dense(
      qat_layer.weight_shadow,
      qat_layer.name,
      delta: qat_layer.delta,
      bias: qat_layer.bias
    )
  end

  @doc """
  Freeze and validate a QAT layer.
  """
  @spec validate_frozen(t()) :: {:ok, Layer.t()} | {:error, term()}
  def validate_frozen(%__MODULE__{} = qat_layer) do
    layer = freeze(qat_layer)

    if is_struct(layer, Layer) and elem(layer.shape, 0) > 0 and elem(layer.shape, 1) > 0 do
      {:ok, layer}
    else
      {:error, :invalid_frozen_layer}
    end
  end
end
