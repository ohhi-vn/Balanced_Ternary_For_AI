# Tasks

## 1. Architecture Presets

- [x] 1.1 Add `:yolo`, `:imagenet`, and `:llm` presets to `Architectures` with classification and transform rules, and verify `Architectures.presets/0` includes them
- [x] 1.2 Implement YOLO sensitivity rules (keep detection head, norms, biases FP16) and verify classification tests pass
- [x] 1.3 Implement ImageNet sensitivity rules (keep classifier head and norms FP16) and verify classification tests pass
- [x] 1.4 Implement LLM sensitivity rules (reuse decoder policy; keep embeddings/norms/LM head FP16) and verify classification tests pass

## 2. Converter Tool Wrappers

- [x] 2.1 Add a YOLO converter entry point that sets arch `:yolo` defaults and validates conv reshape, and verify it produces a `.tbin` on a sample checkpoint
- [x] 2.2 Add an ImageNet converter entry point with arch `:imagenet` defaults and verify output `.tbin` loads
- [x] 2.3 Add an LLM converter entry point for sub-8B sharded checkpoints with parameter-count guard and verify rejection above 8B
- [x] 2.4 Wire the three tools into the CLI as `--profile yolo|imagenet|llm` and verify help text lists them

## 3. Validation and Tests

- [x] 3.1 Add architecture classification tests for the three new presets and verify `mix test` passes
- [x] 3.2 Add pipeline integration tests for each profile on small synthetic/sharded inputs and verify `.tbin` round-trips
- [x] 3.3 Update README and conversion guide with usage examples and verify documented commands run

## 4. Integration Check

- [x] 4.1 Run full test suite and escript build, and verify no regressions

## Workflow follow-up

- Archive the change after review requirements are satisfied.