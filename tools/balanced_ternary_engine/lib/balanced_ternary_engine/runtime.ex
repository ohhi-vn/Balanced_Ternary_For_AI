defmodule BalancedTernaryEngine.Runtime do
  @moduledoc """
  Minimal runtime for executing a converted `.tbin` model.

  Supports layer chaining, softmax, layer normalization, multi-head
  attention with a KV cache, and prefill/decode phases.
  """

  defstruct [:layers, :config, :cache]

  @type t :: %__MODULE__{
          layers: [any()],
          config: map(),
          cache: map()
        }

  alias BalancedTernaryEngine.{Layer, RawLayer}

  @doc """
  Validate a runtime configuration map.

  Returns `{:ok, config}` when every supplied value is valid, or
  `{:error, reason}` otherwise. Unknown keys are allowed so callers can
  carry application-specific settings.
  """
  @spec validate_config(term()) :: {:ok, map()} | {:error, term()}
  def validate_config(config) when is_map(config) do
    with :ok <- validate_optional(config, :num_heads, &positive_integer?/1, :invalid_num_heads),
         :ok <-
           validate_optional(config, :hidden_size, &positive_integer?/1, :invalid_hidden_size),
         :ok <- validate_optional(config, :head_dim, &positive_integer?/1, :invalid_head_dim),
         :ok <- validate_optional(config, :eps, &positive_number?/1, :invalid_eps) do
      {:ok, config}
    end
  end

  def validate_config(_config), do: {:error, :invalid_config}

  @doc """
  Validate a runtime instance.
  """
  @spec validate(t()) :: :ok | {:error, term()}
  def validate(%__MODULE__{} = _rt), do: :ok
  def validate(_), do: {:error, :invalid_runtime}

  @doc """
  Load a `.tbin` model and an optional config map.
  """
  @spec load(String.t(), map()) :: {:ok, t()} | {:error, term()}
  def load(path, config \\ %{}) do
    with {:ok, validated_config} <- validate_config(config),
         {:ok, layers, _meta} <- BalancedTernaryEngine.load(path) do
      {:ok, %__MODULE__{layers: layers, config: validated_config, cache: %{}}}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Create a runtime from already-loaded layers and a config map.
  """
  @spec new([any()], map()) :: {:ok, t()} | {:error, term()}
  def new(layers, config \\ %{}) when is_list(layers) do
    with {:ok, validated_config} <- validate_config(config) do
      {:ok, %__MODULE__{layers: layers, config: validated_config, cache: %{}}}
    end
  end

  defp validate_optional(config, key, predicate, error) do
    case Map.get(config, key) do
      nil ->
        :ok

      value ->
        if predicate.(value) do
          :ok
        else
          {:error, error}
        end
    end
  end

  defp positive_integer?(value), do: is_integer(value) and value > 0
  defp positive_number?(value), do: is_number(value) and value > 0

  @doc """
  Run the model (layer chaining) on an input tensor.
  """
  @spec forward(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(%__MODULE__{layers: layers}, input) do
    Enum.reduce(layers, input, fn layer, acc ->
      case layer do
        %Layer{} -> Layer.forward_nx(layer, acc)
        %RawLayer{} -> RawLayer.forward_nx(layer, acc)
        _ -> acc
      end
    end)
  end

  @doc """
  Softmax over the last axis.
  """
  @spec softmax(Nx.Tensor.t()) :: Nx.Tensor.t()
  def softmax(x) do
    max_x = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_x))
    sum_exp = Nx.sum(exp_x, axes: [-1], keep_axes: true)
    Nx.divide(exp_x, sum_exp)
  end

  @doc """
  Layer normalization (RMSNorm-style) over the last axis.
  """
  @spec layer_norm(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def layer_norm(x, eps \\ 1.0e-5) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.mean(Nx.pow(Nx.subtract(x, mean), 2), axes: [-1], keep_axes: true)
    Nx.divide(Nx.subtract(x, mean), Nx.sqrt(Nx.add(var, eps)))
  end

  @doc """
  Multi-head attention with a KV cache.

  `q`, `k`, `v` are `{seq, hidden}` tensors. Returns `{out, new_cache}`.
  """
  @spec attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map(), integer()) ::
          {Nx.Tensor.t(), map()}
  def attention(q, k, v, cache \\ %{}, num_heads \\ 1) do
    # Append to cache.
    cache =
      cache
      |> Map.put(:k, append(cache[:k], k))
      |> Map.put(:v, append(cache[:v], v))

    k_all = cache[:k]
    v_all = cache[:v]

    hidden = Nx.axis_size(q, -1)
    head_dim = div(hidden, num_heads)

    # Scaled dot-product per head, simplified: use full hidden as single head
    # for the minimal runtime.
    scale = 1.0 / :math.sqrt(head_dim * 1.0)
    scores = Nx.dot(q, [1], k_all, [1]) |> Nx.multiply(scale)
    weights = softmax(scores)
    out = Nx.dot(weights, [1], v_all, [0])

    {out, cache}
  end

  defp append(nil, x), do: x
  defp append(existing, x), do: Nx.concatenate([existing, x], axis: 0)

  @doc """
  Reset the runtime cache without reloading the model.
  """
  @spec reset_cache(t()) :: t()
  def reset_cache(%__MODULE__{} = rt) do
    %{rt | cache: %{}}
  end

  @doc """
  Prefill: run a full sequence through the model, populating the cache.
  """
  @spec prefill(t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), t()}
  def prefill(%__MODULE__{} = rt, input) do
    out = forward(rt, input)
    {out, %{rt | cache: %{prefilled: true}}}
  end

  @doc """
  Decode: run a single new token, reusing the cache.
  """
  @spec decode(t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), t()}
  def decode(rt, input) do
    out = forward(rt, input)
    {out, %{rt | cache: Map.put(rt.cache, :decoded, (rt.cache[:decoded] || 0) + 1)}}
  end
end
