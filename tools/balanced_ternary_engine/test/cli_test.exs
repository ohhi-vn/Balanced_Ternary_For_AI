defmodule BalancedTernaryEngine.CLITest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.CLI

  @tmp System.tmp_dir!()

  test "demo runs" do
    assert :ok =
             CLI.main(["demo", "--rows", "2", "--cols", "2", "--batch", "1", "--delta", "0.5"])
  end

  test "synthetic convert runs" do
    out = Path.join(@tmp, "cli_synthetic.tbin")
    File.rm(out)
    assert :ok = CLI.main(["convert", "--delta", "0.5", "--output", out])
    assert File.exists?(out)
  end

  test "file convert runs" do
    src = Path.join(@tmp, "cli_input.safetensors")
    out = Path.join(@tmp, "cli_file.tbin")
    tensors = %{"fc.weight" => Nx.tensor([[0.8, -0.3], [0.5, -0.2]])}
    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    assert :ok = CLI.main(["convert", "--input", src, "--arch", "generic", "--output", out])
    assert File.exists?(out)
  end

  test "info runs" do
    out = Path.join(@tmp, "cli_info.tbin")
    :ok = CLI.main(["convert", "--delta", "0.5", "--output", out])
    assert :ok = CLI.main(["info", "--model", out])
  end

  test "validate runs" do
    out = Path.join(@tmp, "cli_validate.tbin")
    :ok = CLI.main(["convert", "--delta", "0.5", "--output", out])
    assert :ok = CLI.main(["validate", "--model", out])
  end

  test "qat runs" do
    out = Path.join(@tmp, "cli_qat.tbin")
    assert :ok = CLI.main(["qat", "--epochs", "1", "--output", out])
    assert File.exists?(out)
  end

  test "profile runs" do
    src = Path.join(@tmp, "cli_profile.safetensors")
    out = Path.join(@tmp, "cli_profile.tbin")

    tensors = %{
      "backbone.0.conv.weight" =>
        Nx.Random.uniform(Nx.Random.key(1), -0.5, 0.5, shape: {2, 2, 2, 2}) |> elem(0)
    }

    :ok = BalancedTernaryEngine.Safetensors.write(src, tensors)
    assert :ok = CLI.main(["convert", "--profile", "yolo", "--input", src, "--output", out])
    assert File.exists?(out)
  end
end
