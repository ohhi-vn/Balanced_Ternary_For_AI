# Runtime Guide

```elixir
{:ok, rt} = BalancedTernaryEngine.Runtime.load("model.tbin", %{})
out = BalancedTernaryEngine.Runtime.forward(rt, input)
{out, rt} = BalancedTernaryEngine.Runtime.prefill(rt, input)
{out, rt} = BalancedTernaryEngine.Runtime.decode(rt, input)
```

Reset the cache with `BalancedTernaryEngine.Runtime.reset_cache/1`.