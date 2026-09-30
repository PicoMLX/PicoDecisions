# Measure inference and routing

The `picodecisions` executable runs local checkpoints through the same public
API used by applications. Use a Metal-capable Apple silicon Mac, macOS 15+,
Swift 6.3+, and a full Xcode installation. Download the pinned checkpoint using
the [README](../README.md) first. The executable performs no downloads.

## Run completed-inference benchmarks

```sh
Scripts/run.sh benchmark --model models/laya-multilingual --precision float16 \
  --questions 1,8,20 --batch-size 16 --iterations 20 --warmup 3 \
  --output benchmark-fp16.json
Scripts/run.sh benchmark --model models/laya-multilingual --precision float32 \
  --questions 1,8,20 --batch-size 16 --iterations 20 --warmup 3 \
  --output benchmark-fp32.json
```

Each command launches a fresh process. Workloads repeat the same short order
request with the specified number of independent choice questions; the 20-question
workload crosses the default 16-question GPU batch boundary. Use `--state-file`
to supply a UTF-8 state and inspect reported token counts. Oversized input throws
rather than truncating. This CLI intentionally uses the model's default `.reject`
policy. Reports record the character count and token counts, without copying the
state text.

Timings use Swift's monotonic `ContinuousClock` and await completed `predict`
calls. They include prompt preparation, tokenization, GPU evaluation, calibration,
and conversion to Swift results. JSON serialization is outside the timed region.

| Report field | Meaning |
| --- | --- |
| `loadMilliseconds` | Model loading and initialization after fingerprinting the local files |
| `firstPredictionMilliseconds` | First completed call for that workload, before its warmups |
| `samplesMilliseconds` | Every completed warm prediction, in measurement order |
| `medianMilliseconds` | Middle sample, averaging the two middle samples for even counts |
| `p95Milliseconds` | Nearest-rank 95th percentile; small runs often report their maximum |
| `questionsPerSecond` | Questions per request divided by mean completed-call time |
| `memoryAfterLoad` | MLX active, cached, and peak active allocation bytes after loading |
| Workload `memory` | MLX allocation bytes, with peak reset before that workload's first call |

Fingerprinting reads the weight, encoder config, agent config, tokenizer, and
tokenizer configuration files
before loading. Consequently, loading uses a warm filesystem cache; it does not
measure a cold disk read. First-call times can include kernel initialization;
later workloads also inherit kernels and allocation caches from earlier ones.
Use separate invocations with one `--questions` value when comparing cold shapes.
MLX's peak active allocation is not total process RSS or total device memory.

## Recorded development run

On September 29, 2026, the pinned multilingual checkpoint was measured on an
Apple M1 Max with 32 GiB RAM, macOS 26.6.2, Swift 6.4, Xcode 27, MLX Swift 0.31.6,
and swift-transformers 1.3.4. The Release executable was built with code coverage
and indexing disabled. Each workload used the default 54-character state,
47 tokens per question, three warmup calls, 20 timed calls, and batch size 16.

| Questions per call | FP16 median / p95 (ms) | FP32 median / p95 (ms) |
| --- | --- | --- |
| 1 | 60.59 / 91.89 | 33.91 / 72.77 |
| 8 | 93.28 / 113.53 | 93.85 / 149.26 |
| 20 | 193.74 / 210.30 | 199.48 / 220.57 |

Loading after fingerprinting took 1.59 seconds for FP16 and 1.66 seconds for FP32.
Peak active MLX allocation for the 20-question workload was about 0.97 GiB in
FP16 and 1.94 GiB in FP32. This excludes cached allocation and process RSS.
These are single sequential development runs on a shared machine, with repeated
identical questions and short state. They do not establish a general precision
speed ranking, long-input performance, or production latency. Raw samples and
file fingerprints are retained in [FP16](Measurements/2026-09-29-fp16.json) and
[FP32](Measurements/2026-09-29-fp32.json) reports.

Reports include hardware, OS, build configuration, precision, batch size, and file
SHA-256 hashes. `Scripts/run.sh` builds Release by default. Set
`PICODECISIONS_BUILD_CONFIGURATION=Debug` for development; do not compare Debug
results with Release results. Reuse the built executable for repeated runs to
avoid unnecessary builds. Machine load and thermal state can affect measurements.

## Evaluate candidate routing

```sh
Scripts/run.sh evaluate --model models/laya-multilingual \
  --dataset Evaluation/routing-smoke.json --max-candidates 8 \
  --output routing.json
```

The checked-in dataset is a handwritten development smoke test. Its candidate
rankings are supplied manually, with deliberately poor rankings and one missing
candidate. It exercises the pipeline and documents failure cases; it does not
estimate production accuracy or compare PicoDecisions with Jev.

The recorded FP16 [smoke run](Measurements/2026-09-29-routing-smoke.json) selected
an acceptable answer in 12 of 14 cases, against 5 of 14 for the manually supplied
top-candidate baseline. One miss was an expected tool absent from retrieval; the
other selected `cancel_order` despite an explicit request to do nothing. The
report retains both failures. This small, deliberately challenging candidate
ranking exercise is not evidence that the selector improves real retrieval or
that its probabilities are calibrated. Application authorization and fallback
remain necessary.

For a representative evaluation, supply a held-out dataset from your actual
retrieval stage. Keep candidate order as returned by retrieval. Each JSON case
has `id`, `state`, `expectedToolIDs`, and `candidates`; each candidate has `id`,
`description`, and an optional finite `retrievalScore`. The top-ranked candidate
is the retrieval-only baseline. Scores are metadata and are never fed to the model.
Use an empty `expectedToolIDs` array when no tool should match. Multiple expected
IDs mean any one is an acceptable single recommendation; this is not multi-action
planning. An expected tool may be absent from the candidates to measure retrieval
misses. See [the complete example](../Evaluation/routing-smoke.json).

| Metric | Definition |
| --- | --- |
| `retrievalAccuracy` | Fraction of cases where the first candidate is acceptable; empty candidates predict no match |
| `decisionAccuracy` | Fraction where the model-selected tool is acceptable, or no match is correctly selected |
| `meanCandidateRecall` | Mean fraction of expected IDs present in candidates, across cases with expected tools |
| `retrievalMissCaseCount` | Cases with expected tools but none present in the candidates |
| `falseRejectionRate` | Fraction of cases with at least one acceptable candidate that return no match |
| `falseAcceptanceRate` | Fraction of no-match cases that select a tool |
| `expectedCalibrationError` | Ten equal-width probability bins, weighted absolute difference between selection accuracy and selected-answer probability |

Calibration uses the selected answer's probability, including the no-match
probability, rather than the model's entropy confidence. Empty candidate sets
skip inference and are excluded from calibration. Metrics with no relevant cases
are omitted from JSON. Small datasets cannot establish calibrated confidence.
Calibration labels are relative to the available options: no match is correct
when retrieval supplied no acceptable tool. End-to-end decision accuracy still
counts such a retrieval miss as incorrect when a tool was expected globally.
The report retains each case's selected IDs, distributions, and timing; keep
private evaluation reports out of public commits.

Evaluation timings include the first model call and empty-candidate fast paths,
so use the benchmark command for steady-state performance. Neither the evaluator
nor selector executes tools. The consuming application must collect required
arguments and apply its authorization, fallback, and escalation rules.

## Reuse builds

`PICODECISIONS_DERIVED_DATA_PATH` overrides the ignored `DerivedData` output
directory. `PICODECISIONS_PACKAGE_CACHE_PATH` selects an Xcode package cache.
`PICODECISIONS_BUILD_JOBS` controls compiler parallelism (default: two jobs).
The script disables code coverage and indexing for measurement builds.
Compiler output goes to stderr; stdout contains only JSON, except `--help`.
Trained weights live under the ignored `models` directory. Save private reports
outside the repository or under an ignored local directory.
