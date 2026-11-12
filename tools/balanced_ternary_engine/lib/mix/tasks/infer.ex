defmodule Mix.Tasks.Infer do
  @moduledoc "Run inference on a .tbin model."
  use Mix.Task

  @shortdoc "Run inference on a .tbin model"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [model: :string, input: :string, max_tokens: :integer]
      )

    model = Keyword.get(opts, :model, "model.tbin")
    IO.puts("Loading #{model}...")
    {:ok, layers, _meta} = BalancedTernaryEngine.load(model)
    IO.puts("Loaded #{length(layers)} layers.")
    IO.puts("Inference task is a placeholder; wire a prompt/tokenizer to run generation.")
  end
end
