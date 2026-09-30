# Apply an explicit tool acceptance policy

`ToolDecisionSelector` recommends a tool or no match from a retrieved candidate
set. Its default behavior is unchanged: it returns the model's selected answer
and probability distribution without applying a threshold. An application can
apply `ToolDecisionAcceptancePolicy` afterward when it wants to defer uncertain
recommendations.

```swift
import PicoDecisions

let selector = try ToolDecisionSelector(model: model)
let selection = try await selector.select(query: request, candidates: retrievedTools)

// Illustrative thresholds; choose application thresholds on held-out requests.
let policy = try ToolDecisionAcceptancePolicy(minimumProbability: 0.8, minimumMargin: 0.1)
let disposition = policy.evaluate(selection)

switch disposition.status {
case .acceptedTool:
    let candidate = disposition.selection.selectedCandidate!
    // Route candidate.id to the application's argument and authorization checks.
case .acceptedNoMatch:
    // Use the application's no-match response or broader retrieval fallback.
    break
case .abstained:
    // Ask for clarification, use a fallback, or escalate under application policy.
    // disposition.reasons explains which gate failed.
    break
}
```

Both thresholds must be supplied explicitly and must be finite values in
`0...1`. Zero disables that gate; setting both to zero preserves every valid raw
recommendation. Equality passes. This policy applies the same gates to tool
answers and inferred no-match answers.

| Field | Meaning |
| --- | --- |
| `minimumProbability` | Minimum raw probability of the model-selected answer |
| `minimumMargin` | Minimum gap between that answer and its strongest rival |
| `selectedProbability` | Probability of the selected tool or selected no-match answer |
| `probabilityMargin` | Selected probability minus the strongest competing probability, including no match |
| `selection` | Original recommendation, probabilities, retrieval metadata, and model metadata |
| `reasons` | Failed gates, or `invalidSelection` for malformed input |

A tool selected with probability `0.55` against no match at `0.40` has a margin
of `0.15`, regardless of how low the other tools' probabilities are. Negative
margins are retained if an injected model selected an answer other than its
highest-probability option. A positive margin gate abstains on such answers;
a disabled gate leaves the raw answer intact.

Abstention preserves the original answer; it does not convert an uncertain tool
recommendation into a no-match prediction. Empty candidate sets are different:
the selector skips inference and deterministically returns no match. The policy
accepts that result without creating probabilities or confidence. Its derived
probability and margin are nil. Exclude those cases when measuring model coverage
or model abstention rates.

The policy reads selected-answer probabilities, including no match. It does not
use retrieval scores, entropy confidence, or the separate action head. Those
values remain in the original selection. Accepted recommendations still require
the application's argument collection, authorization, execution, and fallback
rules. Applying a policy does not rerun inference or change model quality.

Inspect `selection.inputDiagnostics` when requests or tool descriptions are
long. Laya may shorten instructions and options even with its state input policy
set to `.reject`. The selector and policy preserve those counts so the
application can show a warning or defer a result with lost criteria. See
[prompt diagnostics](InputDiagnostics.md).

## Evaluate the tradeoff

The [recorded development smoke run](Measurements/2026-09-29-routing-smoke.json)
selected cancellation despite an explicit request to do nothing. The selected
tool's probability was about `0.5316`, against no match at `0.4587`: a margin of
about `0.0729`. A caller's thresholds can defer that recommendation. The model's
incorrect answer remains in the disposition for evaluation and review.

Thresholds can also defer correct answers. For example, the same run correctly
selected `find_order` for the French request with probability about `0.7485`;
the illustrative `0.8` probability threshold above would abstain. All inferred
cases in this report had action probability `1`, so that head supplied no
discrimination in this run.

Choose thresholds on held-out data from the consuming application's retrieval
stage. Keep raw accuracy separate from acceptance coverage, accepted-answer
accuracy, and uncertain abstention. Compare candidate-relative labels with
end-to-end labels so retrieval misses remain visible. An abstention is neither
an accepted answer nor a correct no-match prediction. The handwritten smoke
dataset demonstrates behavior and cannot establish production thresholds or
out-of-distribution performance.
