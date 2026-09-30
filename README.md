# PicoDecisions

Typed, on-device decisions for Swift: classification, routing, ordinal scoring,
and boolean judgments. Models answer questions without generating text.

The first backend ports Laya to MLX Swift. It runs entirely in-process and loads
local safetensors checkpoints; Python is only used to generate development
fixtures. The public API remains provisional.

## Products

| Product | Responsibility |
| --- | --- |
| `PicoDecisions` | Backend-independent requests, typed results, and `DecisionModel` |
| `PicoDecisionsMLX` | Laya inference, tokenization, checkpoint loading, and calibration |

The core target has no runtime dependencies. The MLX target uses MLX Swift and
swift-transformers' Tokenizers product. SwiftPM still resolves the package's
backend dependencies. Core ML is planned, but not implemented.

Requires Swift 6.2+, macOS 15+ or iOS 18+, and Apple silicon for MLX. Validation
currently covers macOS; physical iOS device validation remains outstanding.

## Use a local checkpoint

Add this package as a local Swift package dependency and link `PicoDecisionsMLX`.
Download the validated multilingual checkpoint separately, for example with the
Hugging Face CLI:

```sh
hf download convaiinnovations/laya-multilingual \
  --revision 052592a15d198d9ad47da779604259b10b47b7aa \
  --include 'model.safetensors' 'encoder/config.json' 'rl_agent_config.json' 'tokenizer/*' \
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
serialized JSON). Custom false/true boolean rubrics are not exposed in this API.

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
plain `swift test` does not compile the Metal library needed for inference tests.

## Integration and next steps

PicoDecisions evaluates questions; consuming applications own tool retrieval and
execution. SmartToolSelection can retrieve a bounded candidate set and pass its
IDs/descriptions to `DecisionModel`, with an explicit no-match choice. This adapter
and end-to-end retrieval evaluation remain future work. See the
[implementation plan](Docs/ImplementationPlan.md).

## Attribution

Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Inference is adapted from
[original Laya](https://github.com/NandhaKishorM/laya) and
[laya-mlx](https://github.com/mizorewww/laya-mlx); real-model validation uses
[laya-coreml's PyTorch reference](https://github.com/mizorewww/laya-coreml).
This is an independent project. Trained weights are downloaded separately and
retain their authors' licenses. Included test weights are synthetic random data.
