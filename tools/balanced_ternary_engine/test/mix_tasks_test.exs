defmodule BalancedTernaryEngine.MixTasksTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.Safetensors
  alias Mix.Tasks.{Analyze, Convert, Infer}

  @tmp System.tmp_dir!()

  test "Convert task no input" do
    Convert.run([])
  end

  test "Convert task failure" do
    assert_raise Mix.Error, fn ->
      Convert.run([
        "--input",
        "/nope.safetensors",
        "--output",
        Path.join(@tmp, "x.tbin")
      ])
    end
  end

  test "Convert task runs on a file" do
    src = Path.join(@tmp, "mix_convert.safetensors")
    out = Path.join(@tmp, "mix_convert.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = Safetensors.write(src, tensors)
    Convert.run(["--input", src, "--output", out, "--arch", "generic"])
    assert File.exists?(out)
  end

  test "Analyze task runs" do
    src = Path.join(@tmp, "mix_analyze.safetensors")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = Safetensors.write(src, tensors)
    Analyze.run(["--input", src])
  end

  test "Infer task runs" do
    src = Path.join(@tmp, "mix_infer_in.safetensors")
    out = Path.join(@tmp, "mix_infer.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = Safetensors.write(src, tensors)
    {:ok, _} = BalancedTernaryEngine.Pipeline.run(src, out, arch: :generic)
    Infer.run(["--model", out])
  end
end
