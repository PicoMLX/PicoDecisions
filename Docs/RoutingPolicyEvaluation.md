# Evaluating an optional routing acceptance policy

`evaluate` can apply caller-chosen probability and margin thresholds after the
model's raw recommendation. No thresholds are enabled by default. This is a
development tool for measuring coverage and errors, not a production calibration
procedure. A high model probability can accompany an incorrect recommendation.

```sh
Scripts/run.sh evaluate --model /path/to/checkpoint \
  --dataset Evaluation/routing-smoke.json \
  --minimum-probability 0.8 --minimum-margin 0.1 \
  --output /tmp/routing-policy.json
```

Both flags accept finite numbers in `0...1` and are available only for
`evaluate`. If neither flag is supplied, the acceptance policy is disabled.
Supplying one flag sets the other threshold to zero. A zero threshold disables
that individual gate; explicitly supplying zero still enables policy reporting.
Positive thresholds accept equality. The margin is the probability of the raw
selected answer minus the strongest competing answer, including no match. It is
reported as computed, including a negative value if the raw selected answer is
not the most probable answer. Entropy confidence, retrieval scores, and action
probability are not used by this policy.

The existing `selectedID`, distributions, confidence, raw decision accuracy,
retrieval metrics, and ECE retain their original meanings. Schema version 2 adds
optional `policyConfiguration`, `policyMetrics`, and per-outcome `policy` fields.
They are omitted when the policy is disabled. Configuration records effective
`minimumProbability` and `minimumMargin`, with each source identified as
`commandLine` or `defaultZero`.

Each outcome's policy reports `acceptedTool`, `acceptedNoMatch`, or `abstained`,
its reasons, and the raw selected probability and margin when available.
Abstention leaves the raw selection visible and is never scored as a correct
no-match answer. `acceptedNoMatch` is a separate accepted recommendation. Empty
candidate sets skip model inference, have no distribution or derived scores,
and are excluded from all policy denominators even though their deterministic
outcome is no match.

Policy metrics use the following denominators:

| Field | Meaning |
| --- | --- |
| `inferredCaseCount` | Cases with at least one supplied candidate |
| `coverage` | Accepted inferred cases / inferred cases |
| `abstentionRate` | Abstained inferred cases / inferred cases |
| `decisionAccuracy` | Accepted end-to-end correct answers / inferred cases; abstentions earn no credit |
| `candidateRelativeAccuracy` | Accepted answers correct for the supplied candidate set / inferred cases |
| `acceptedAccuracy` | End-to-end correct answers / accepted inferred cases |
| `acceptedCandidateRelativeAccuracy` | Answers correct for the supplied candidate set / accepted inferred cases |
| `falseAcceptanceRate` | Accepted tools on labeled no-match cases / inferred labeled no-match cases |

Candidate-relative correctness permits accepted no match when none of the
acceptable labeled tools was retrieved. End-to-end correctness still requires
an acceptable labeled tool in those retrieval-miss cases. Counts distinguish
accepted tools, accepted no-match answers, and abstentions. A rate with a zero
denominator is omitted from JSON rather than reported as zero.

The raw `metrics` object also adds `inferredCaseCount`,
`inferredNoMatchCaseCount`, and `inferredFalseAcceptanceRate` independently of
policy activation. These expose model behavior separately from deterministic
empty-candidate outcomes. For example, one false acceptance across two inferred
no-match cases is `1/2`; adding a correctly empty candidate set does not improve
that inferred rate to `1/3`. The original all-case `falseAcceptanceRate` remains
available for compatibility.

The handwritten routing smoke dataset is intended for development and regression
checks. It cannot establish a production threshold or justify a calibrated
accuracy claim. Choose thresholds using a separate representative validation
dataset and evaluate the chosen values on held-out data. Publish both coverage
and accepted-only errors; an increased accepted-only accuracy can simply reflect
rejecting more cases. Record dataset/checkpoint hashes and the explicit thresholds
with each measurement.

## Recorded development check

The [September 30 smoke report](Measurements/2026-09-30-routing-policy-smoke.json)
uses explicit thresholds of `0.8` probability and `0.15` margin. It retains raw
accuracy of 12/14. Among the 13 cases that ran inference, the policy accepted 11
and abstained on two: the incorrect cancellation and the correct French lookup.
Accepted end-to-end accuracy was 10/11; candidate-relative accuracy was 11/11.
The accepted retrieval miss remains an end-to-end error.

This check used the Debug executable on the same M1 Max with the pinned FP16
checkpoint. Its timings are not comparable to the recorded Release benchmarks.
The handwritten cases demonstrate the coverage tradeoff; these thresholds are
illustrative and have not been selected on representative held-out data.
