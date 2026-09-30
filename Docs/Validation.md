# Validation

Validated on an Apple silicon Mac using Swift 6.4, Xcode's macOS SDK 27, MLX Swift
0.31.6, and swift-transformers 1.3.4. `Package.resolved` records the dependency graph.

## Results

The Xcode package test suite passes, including the optional trained-checkpoint
suite. Tests cover request validation, configuration and calibration, strict
weight loading, prompt formatting, attention-window boundaries, padded queries,
repeated calls, mixed question batches, and both inference precisions.
The expanded suite also covers concurrent callers, canceled load/predict
calls, malformed tokenizer/config/checkpoint files, exact state capacity,
255/256-option limits, the candidate selector, and evaluation metric definitions.

The September 29, 2026 verification passed 16 core, eight CLI, and 26 backend
test functions, including parameterized cases. The same tests also passed through
SwiftPM on the Swift 6.4 / Xcode 27 toolchain, which generated the Metal bundle.
Hosted core/CLI checks also passed on Swift 6.3.3 after the initial milestone.
Physical iOS inference still requires independent checks.

The trained multilingual suite compares 16 cases /
63 questions against an existing PyTorch FP32 reference. It includes English, Chinese, German, French,
Spanish, Hindi, Japanese, Russian, empty state, long-input truncation, conversation
JSON, literal mask text, 20 questions in one request, and 20 choice options.
All token IDs, marker positions, question types, token counts, and selected
choices match. Public results are tested in batches of eight; the raw network
comparison uses each reference case's full question batch.

The structured-criteria case covers JSON instruction/choice/score descriptions
and a custom true boolean criterion with the default false criterion. The expanded
fixture passed a fresh FP32/FP16 checkpoint-suite run on September 30, 2026; the
largest numerical differences below remained unchanged.

| Comparison | Largest absolute difference observed |
| --- | --- |
| FP32 raw decision logits | 0.00005341 |
| FP32 raw action logits | 0.00134278 |
| FP32 public numerical outputs | 0.00004994 |
| FP16 public numerical outputs | 0.00152819 |

Public numerical outputs include choice/score probabilities, expected score,
boolean probability, confidence, and act probability. The original public JSON
rounds values to four decimals; Swift keeps full precision. Action logits can
have large magnitudes, so their absolute error is separate from probability error.
Test tolerances allow small numerical differences between supported Metal systems:
0.001 for FP32 decision logits, 0.01 for action logits, 0.0002 for FP32 public
outputs, and 0.005 for FP16 public outputs.

These are implementation-parity checks, not evidence of task accuracy or measured
latency. All tested trained-model choice labels match in both precisions. The tiny
random model has near-tied options, so its FP16 test checks probability agreement
rather than requiring an unchanged winning label. A Release benchmark/evaluation
CLI is now available; see [measurement scope and reproduction](Benchmarks.md).
Physical iOS inference, quantization, Core ML, and the
English/typed checkpoints remain unvalidated.

## Reference provenance

- Python MLX implementation: `mizorewww/laya-mlx` at
  `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`.
- Trained-model reference: `benchmarks/results/reference.json` from
  `mizorewww/laya-coreml` at `12b7501583c7f03a6b2e49ebe118a2c6302505b9`.
- Original Laya revision used by that reference:
  `6a5819129eb220570792e417e49723d697efd76f`.
- Checkpoint: `convaiinnovations/laya-multilingual` at
  `052592a15d198d9ad47da779604259b10b47b7aa`.
- Original `model.safetensors` SHA-256:
  `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`.

The normalized fixture preserves original outputs and precomputed token IDs. It
serializes structured state, instructions, and criteria using Python's original
JSON representation and turns question dictionaries into ordered arrays. Custom
boolean descriptions retain false/true labels and use defaults for omitted sides.
No trained model weights are included in the repository.

## Reproduce

Run the commands in the [README](../README.md) to execute default or optional
trained-checkpoint tests, or use `Scripts/test.sh inference` and
`PICODECISIONS_LAYA_MODEL=/absolute/path/to/model Scripts/test.sh checkpoint`.
Tests do not download models. With Xcode, the
`TEST_RUNNER_PICODECISIONS_LAYA_MODEL` environment variable is forwarded as
`PICODECISIONS_LAYA_MODEL` to the test process. Without that variable, full-checkpoint
tests are skipped.

The small synthetic fixture is checked in and can be regenerated on Apple silicon
with a checkout of the pinned Python implementation:

```sh
python Scripts/generate_fixtures.py /path/to/laya-mlx
```

The generator requires `mlx`, `numpy`, `tokenizers`, `safetensors`, and
`huggingface_hub`; it was run with MLX 0.31.2 and NumPy 2.5.3. Its safetensors contain
small deterministic random weights (seed 20260920), not trained Laya parameters.
The synthetic test checks exact tokenization and raw FP32 logits within 0.00003.

To recreate the normalized real-checkpoint reference:

```sh
python Scripts/import_reference.py /path/to/laya-coreml/benchmarks/results/reference.json
```

Use the pinned repositories above when regenerating either fixture. See
[NOTICE](../NOTICE) for attribution.
