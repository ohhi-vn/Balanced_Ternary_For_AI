defmodule Mix.Tasks.Analyze do
  @moduledoc "Analyze a .tbin model."
  use Mix.Task

  @shortdoc "Analyze a .tbin model"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [model: :string, show_sparsity: :boolean, show_compression: :boolean]
      )

    model = Keyword.get(opts, :model, "model.tbin")
    {:ok, layers, _meta} = BalancedTernaryEngine.load(model)
    stats = BalancedTernaryEngine.stats(layers)

    if Keyword.get(opts, :show_sparsity, true) do
      IO.puts("Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
    end

    if Keyword.get(opts, :show_compression, true) do
      IO.puts("Compression: #{Float.round(stats.compression_ratio, 1)}x")
      IO.puts("Packed: #{Float.round(stats.packed_size_mb, 2)} MB")
    end
  end
end
