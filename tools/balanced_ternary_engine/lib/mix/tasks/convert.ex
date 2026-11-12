defmodule Mix.Tasks.Convert do
  @moduledoc "Convert a pretrained model to balanced ternary format."
  use Mix.Task

  @shortdoc "Convert a model to .tbin"

  @impl Mix.Task
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          input: :string,
          output: :string,
          arch: :string,
          mode: :string,
          delta: :float,
          sparsity: :float,
          sparsity_target: :float,
          clip: :float,
          calibration_data: :string,
          target_similarity: :float
        ]
      )

    arch = opts |> Keyword.get(:arch, "generic") |> String.to_atom()
    mode = opts |> Keyword.get(:mode, "twn") |> String.to_atom()
    input = Keyword.get(opts, :input)

    output =
      Keyword.get(
        opts,
        :output,
        if(input,
          do: String.replace(Path.basename(input), Path.extname(input), ".tbin"),
          else: "model.tbin"
        )
      )

    cli_opts =
      [arch: arch, mode: mode]
      |> maybe_put(:delta, Keyword.get(opts, :delta))
      |> maybe_put(:sparsity, Keyword.get(opts, :sparsity))
      |> maybe_put(:sparsity_target, Keyword.get(opts, :sparsity_target))
      |> maybe_put(:clip, Keyword.get(opts, :clip))
      |> maybe_put(:calibration_data, Keyword.get(opts, :calibration_data))
      |> maybe_put(:target_similarity, Keyword.get(opts, :target_similarity))

    if input do
      case BalancedTernaryEngine.Pipeline.run(input, output, cli_opts) do
        {:ok, summary} ->
          IO.puts("Converted #{summary.ternary_layers |> length()} ternary layers to #{output}")

        {:error, reason} ->
          raise Mix.Error, message: "Conversion failed: #{inspect(reason)}"
      end
    else
      # synthetic path via CLI helper is not exposed; run a minimal conversion
      IO.puts("No --input given; use the escript CLI for synthetic models.")
    end
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
