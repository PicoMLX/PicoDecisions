# PicoDecisions

Typed, on-device decisions for Swift: classification, routing, ordinal scoring,
and boolean judgments. Models answer questions without generating text.

The first backend ports Laya to MLX Swift. It runs entirely in-process and loads
local safetensors checkpoints; Python is only used to generate development
fixtures. The public API remains provisional.

This is an experimental runtime. Model parity, routing accuracy, and performance
are measured separately; see [benchmarks and evaluation](Docs/Benchmarks.md).

## Products

| Product | Responsibility |
| --- | --- |
| `PicoDecisions` | Backend-independent requests, typed results, and `DecisionModel` |
| `PicoDecisionsMLX` | Laya inference, tokenization, checkpoint loading, and calibration |
| `picodecisions` | Local demo, completed-inference benchmarks, and routing evaluation |

The core target has no runtime dependencies. The MLX target uses MLX Swift and
swift-transformers' Tokenizers product. SwiftPM still resolves the package's
backend dependencies. Core ML is planned, but not implemented.

Requires Swift 6.3+, macOS 15+ or iOS 18+, and Apple silicon for MLX. The pinned
MLX Swift dependency requires Swift 6.3, including when resolving core-only builds.
Validation currently covers macOS; physical iOS device validation remains outstanding.

## Use a local checkpoint

Add [PicoMLX/PicoDecisions](https://github.com/PicoMLX/PicoDecisions) as a Swift
package dependency and link `PicoDecisionsMLX`. The provisional API is available
on `main`; pin a commit when integrating it.
Download the validated multilingual checkpoint separately, for example with the
Hugging Face CLI:

```sh
hf download convaiinnovations/laya-multilingual \
  model.safetensors encoder/config.json rl_agent_config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  --revision 052592a15d198d9ad47da779604259b10b47b7aa \
  --local-dir models/laya-multilingual
```

The runtime reads this directory without downloading files:

```swift
import Foundation
import PicoDecisions
import PicoDecisionsMLX

let model = try await LayaModel.load(
    from: URL(filePath: "/absolute/path/to/models/laya-multilingual"),
    precision: .float16
)

let results = try await model.predict(DecisionRequest(
    state: "Find the order I placed yesterday; don't cancel it.",
    questions: [
        DecisionQuestion(
            id: "next_action",
            instructions: "Which action best matches the request?",
            kind: .choice(options: [
                DecisionOption(id: "find_order", description: "Look up an existing order"),
                DecisionOption(id: "cancel_order", description: "Cancel an existing order"),
                DecisionOption(id: "none", description: "Neither action fits")
            ])
        ),
        DecisionQuestion(id: "urgency", instructions: "How urgent is the request?",
                         kind: .score(levels: ["routine", "soon", "immediate"])),
        DecisionQuestion(id: "cancel", instructions: "Does the user request cancellation?",
                         kind: .boolean)
    ]
))

for result in results {
    print(result.id, result.answer, result.actProbability as Any)
}
```

Results retain question and option order. Choice answers include the selected ID
and option probabilities. Scores are expected **zero-based level indices** with
the full level distribution. Booleans return the probability of true. Each result
also includes model confidence, act probability, and the input token count.

Laya's choice/score confidence is normalized entropy confidence; boolean confidence
is the larger of the true/false probabilities. Act probability is the separate
model action head's probability for acting rather than escalating. These values
are model outputs, not authorization checks or guarantees of correctness. Outputs
retain precision instead of rounding to upstream Python's four decimal places.

`LayaModel` is an actor. It owns its weights and serializes inference, processes
questions in bounded batches (default 16), and checks cancellation between batches.
FP16 and FP32 are supported; quantized checkpoints are not yet supported. Original
Laya parameter names and converted laya-mlx parameter names are accepted with strict
name and shape validation. Only the multilingual trained checkpoint has been validated.

## Input limits

The checkpoint determines context length (1,024 tokens for the validated model).
By default, an oversized state throws `DecisionError.capacityExceeded`. Pass
`inputPolicy: .truncateState` when loading to match upstream Laya's behavior of
retaining the beginning of the state that fits each question.

Question instructions and option descriptions use Laya's own prefix budgets and
may be shortened even with `.reject`. No option may be silently dropped. The
backend accepts at most 255 options, subject to the total context capacity, and
replaces literal mask-token text as upstream does. Request validation rejects
empty option sets, empty score rubrics, blank instructions, and duplicate IDs.
Empty state and zero-question requests are supported.

Supply structured state and criterion descriptions as strings (for example,
serialized JSON). Boolean questions can override either or both false/true
descriptions; the omitted side retains Laya's default description:

```swift
let refundQuestion = DecisionQuestion(
    id: "refund",
    instructions: "Does the request ask for a refund?",
    kind: .boolean,
    booleanCriteria: .init(trueDescription: "The user explicitly requests money back")
)
```

Criteria always keep false then true order, and the result remains the probability
of true. Custom descriptions must be nonblank and belong to a boolean question.
They use the same token budgets and mask-token sanitization as other options.

## Validation

Use Xcode to build the Metal shaders and run the package tests on a Mac:

```sh
xcodebuild test -scheme PicoDecisions-Package \
  -destination 'platform=macOS,arch=arm64'
```

The default suite includes a small deterministic Python-generated model and needs
no downloaded Laya weights. To enable tests against the pinned real checkpoint:

```sh
TEST_RUNNER_PICODECISIONS_LAYA_MODEL="$PWD/models/laya-multilingual" \
  xcodebuild test -scheme PicoDecisions-Package \
  -destination 'platform=macOS,arch=arm64'
```

See [validation details](Docs/Validation.md) for provenance, numerical comparisons,
and fixture regeneration. `swift build --target PicoDecisions` builds the core;
`Scripts/test.sh core` runs core API and CLI checks through SwiftPM. Use the
Xcode-backed inference and checkpoint modes for repeatable Metal validation.

## Integration and next steps

PicoDecisions evaluates questions; consuming applications own tool retrieval and
execution. `ToolDecisionSelector` evaluates a bounded candidate set with an
explicit no-match choice:

```swift
let selector = try ToolDecisionSelector(model: model, maximumCandidates: 8)
let selection = try await selector.select(
    query: "Find yesterday's order; do not cancel it.",
    candidates: [
        ToolDecisionCandidate(id: "find_order", description: "Look up an existing order"),
        ToolDecisionCandidate(id: "cancel_order", description: "Cancel an existing order")
    ]
)
print(selection.selectedCandidateID as Any) // nil means no matching tool
```

The selector preserves candidate IDs and retrieval scores, validates model
responses, and skips inference for an empty candidate set. Retrieval scores are
metadata, separate from decision probabilities. It recommends one tool; callers
own thresholds, argument collection, authorization, execution, and fallback.
SmartToolSelection/PicoCore can supply candidates, but consuming-application
integration and representative retrieval evaluation remain future work.
See the [implementation plan](Docs/ImplementationPlan.md).

Applications can apply `ToolDecisionAcceptancePolicy` after selection to defer
uncertain recommendations using explicit probability and margin thresholds.
It preserves the raw answer and distinguishes abstention from no match; it
does not authorize execution. See the [routing policy guide](Docs/ToolRouting.md).

## Run the demo and measurements

On an Apple silicon Mac with Xcode and its Metal compiler, run:

```sh
Scripts/run.sh demo --model models/laya-multilingual
Scripts/run.sh benchmark --model models/laya-multilingual --precision float16 \
  --questions 1,8,20 --iterations 20 --output benchmark.json
Scripts/run.sh evaluate --model models/laya-multilingual \
  --dataset Evaluation/routing-smoke.json --output evaluation.json
```

The script builds the executable in Release mode using Xcode, which compiles the
required Metal library. Results are JSON. Benchmarks include tokenization and
result processing, separate first-call and warm timings, and report MLX memory.
The included routing dataset is a small handwritten development smoke test with
manually ranked candidates. Use your own labeled requests before making accuracy
or calibration claims. See [measurement details](Docs/Benchmarks.md).

`evaluate` also supports opt-in `--minimum-probability` and `--minimum-margin`
thresholds. Its policy report preserves raw predictions and measures acceptance
coverage, abstention, and accepted-answer errors. See
[policy evaluation](Docs/RoutingPolicyEvaluation.md).

For repeatable validation, use `Scripts/test.sh core`, `Scripts/test.sh inference`,
or `PICODECISIONS_LAYA_MODEL="$PWD/models/laya-multilingual" Scripts/test.sh checkpoint`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for CI and environment details.

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Inference is adapted from
[original Laya](https://github.com/NandhaKishorM/laya) and
[laya-mlx](https://github.com/mizorewww/laya-mlx); real-model validation uses
[laya-coreml's PyTorch reference](https://github.com/mizorewww/laya-coreml).
This is an independent project. Trained weights are downloaded separately and
retain their authors' licenses. Included test weights are synthetic random data.
