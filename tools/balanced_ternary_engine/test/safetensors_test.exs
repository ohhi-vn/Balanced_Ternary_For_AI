defmodule BalancedTernaryEngine.SafetensorsTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.Safetensors

  @tmp System.tmp_dir!()

  describe "write/3 + read/1 round-trip" do
    test "f32 tensors survive a write/read cycle" do
      path = Path.join(@tmp, "st_f32.safetensors")
      tensors = %{"layer.weight" => Nx.tensor([[1.5, -2.25], [0.0, 3.75]])}

      :ok = Safetensors.write(path, tensors)
      {:ok, loaded, _meta} = Safetensors.read(path)

      assert Nx.to_flat_list(loaded["layer.weight"]) == Nx.to_flat_list(tensors["layer.weight"])
      assert Nx.shape(loaded["layer.weight"]) == {2, 2}
    end

    test "f16, bf16 and i64 tensors are normalized to f32/s64" do
      path = Path.join(@tmp, "st_mixed.safetensors")

      f16 = Nx.tensor([1.0, 2.0, -0.5], type: :f16)
      bf16 = Nx.tensor([0.25, -1.5], type: :bf16)
      i64 = Nx.tensor([1, 2, 3], type: :s64)

      :ok = Safetensors.write(path, %{"a" => f16, "b" => bf16, "c" => i64})
      {:ok, loaded, _meta} = Safetensors.read(path)

      assert Nx.type(loaded["a"]) == {:f, 32}
      assert Nx.to_flat_list(loaded["a"]) == [1.0, 2.0, -0.5]

      assert Nx.type(loaded["b"]) == {:f, 32}
      assert_in_delta Nx.to_number(loaded["b"][0]), 0.25, 1.0e-6

      assert Nx.type(loaded["c"]) == {:s, 64}
      assert Nx.to_flat_list(loaded["c"]) == [1, 2, 3]
    end

    test "metadata is preserved under __metadata__" do
      path = Path.join(@tmp, "st_meta.safetensors")
      metadata = %{"format" => "pt", "source" => "test"}

      :ok = Safetensors.write(path, %{"w" => Nx.tensor([1.0])}, metadata)
      {:ok, _tensors, meta} = Safetensors.read(path)

      assert meta == metadata
    end

    test "multiple tensors with padding stay aligned" do
      path = Path.join(@tmp, "st_multi.safetensors")

      # Odd sizes force header/data padding paths
      tensors = %{
        "one" => Nx.tensor([1.0]),
        "two" => Nx.tensor([1.0, 2.0]),
        "three" => Nx.tensor([[[1.0, 2.0], [3.0, 4.0]]])
      }

      :ok = Safetensors.write(path, tensors)
      {:ok, loaded, _meta} = Safetensors.read(path)

      assert map_size(loaded) == 3

      Enum.each(tensors, fn {name, t} ->
        assert Nx.shape(loaded[name]) == Nx.shape(t)
        assert Nx.to_flat_list(loaded[name]) == Nx.to_flat_list(t)
      end)
    end
  end

  test "reading a non-safetensors file returns an error" do
    path = Path.join(@tmp, "st_bad.safetensors")
    File.write!(path, "garbage data that is definitely not safetensors")

    assert {:error, _reason} = Safetensors.read(path)
  end
end

defmodule BalancedTernaryEngine.SafetensorsShardTest do
  use ExUnit.Case, async: true

  alias BalancedTernaryEngine.Safetensors

  @tmp System.tmp_dir!()

  test "read_sharded merges tensors from multiple shards" do
    w1 = Nx.tensor([[1.0, 2.0]])
    w2 = Nx.tensor([[3.0, 4.0]])
    :ok = Safetensors.write(Path.join(@tmp, "shard_a.safetensors"), %{"a" => w1})
    :ok = Safetensors.write(Path.join(@tmp, "shard_b.safetensors"), %{"b" => w2})

    index = %{
      "metadata" => %{},
      "weight_map" => %{"a" => "shard_a.safetensors", "b" => "shard_b.safetensors"}
    }

    File.write!(Path.join(@tmp, "model.safetensors.index.json"), Jason.encode!(index))

    {:ok, tensors, _meta} =
      Safetensors.read_sharded(Path.join(@tmp, "model.safetensors.index.json"))

    assert Map.has_key?(tensors, "a")
    assert Map.has_key?(tensors, "b")
  end

  test "decode returns error on unknown dtype" do
    # Build a minimal safetensors with an unknown dtype by hand.
    header = %{"x" => %{"dtype" => "F99", "shape" => [1], "data_offsets" => [0, 4]}}
    header_json = Jason.encode!(header)
    pad = rem(8 - rem(byte_size(header_json), 8), 8)
    header_padded = header_json <> String.duplicate(" ", pad)
    binary = <<byte_size(header_padded)::64-little>> <> header_padded <> <<0.0::float-32>>

    assert {:error, _} = Safetensors.decode(binary)
  end
end
