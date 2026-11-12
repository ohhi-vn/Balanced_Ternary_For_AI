defmodule BalancedTernaryEngine.MixProject do
  use Mix.Project

  def project do
    [
      app: :balanced_ternary_engine,
      version: "0.1.0-dev",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: escript(),
      test_coverage: [tool: ExCoveralls],
      dialyzer: [plt_add_apps: [:mix]],
      description: "Convert neural network weights to balanced ternary and run them at runtime.",
      package: package(),
      docs: docs(),
      elixirc_paths: elixirc_paths(Mix.env()),
      aliases: aliases(),
      test_coverage: [
        tool: ExCoveralls,
        summary: [threshold: 80],
        ignore_modules: [],
        # Floor of the decision-point (branch) coverage, reported by
        # RealtimeEngine.Test.BranchCoverage at the end of a coverage run.
        branch: [threshold: 100]
      ]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:excoveralls, "~> 0.18", only: :test},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_dna, "~> 1.5", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      name: "balanced_ternary_engine",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/ohhi-vn/Balanced_Ternary_For_AI"},
      files: ["lib", "guides", "mix.exs", "README.md", "LICENSE", "CHANGELOG.md"],
      maintainers: ["Ohhi"]
    ]
  end

  defp docs do
    [
      main: "BalancedTernaryEngine",
      extras: ["README.md", "CHANGELOG.md", "guides/conversion.md", "guides/runtime.md"],
      groups_for_extras: [
        Guides: ["guides/conversion.md", "guides/runtime.md"]
      ]
    ]
  end

  defp escript do
    [main_module: BalancedTernaryEngine.CLI]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp aliases do
    [
      # Testing & Coverage
      coveralls: ["test --cover", "coveralls.html"],
      # Code Quality
      quality: ["format --check-formatted", "credo --strict", "dialyzer"]
    ]
  end
end
