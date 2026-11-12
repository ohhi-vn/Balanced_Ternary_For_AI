defmodule BalancedTernaryEngine.ExporterTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.{Exporter, Layer}

  @tmp System.tmp_dir!()

  test "format_version reports v2 for exported file" do
    w = Nx.tensor([[0.8, -0.3], [0.5, -0.2]])
    layer = Layer.from_dense(w, "fc", delta: 0.5)
    path = Path.join(@tmp, "fmt_v2.tbin")
    :ok = Exporter.export([layer], path)
    assert {:ok, 2} = Exporter.format_version(path)
  end

  test "format_version rejects invalid file" do
    path = Path.join(@tmp, "fmt_bad.tbin")
    File.write!(path, "not a tbin")
    assert {:error, _} = Exporter.format_version(path)
  end
end
