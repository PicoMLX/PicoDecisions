# Implementation plan

## 1. MLX Swift backend — implemented

The initial implementation below is complete for local FP16/FP32 multilingual
checkpoints. `PicoDecisionsMLX` uses MLX Swift and swift-transformers Tokenizers.
Network downloads and quantization are not implemented.

Add a separate `PicoDecisionsMLX` product depending on `PicoDecisions`, MLX Swift,
and the tokenizer/downloader interfaces selected during implementation.

Start with the multilingual Laya checkpoint in FP32 and FP16. Port ModernBERT /
mmBERT, question-type embeddings, the decision Transformer, option scoring, and
the action head. Preserve configuration-driven calibration and expose action
probabilities and token usage when extending the provisional result API.

Load the existing safetensors with strict name/shape validation. Support the
checkpoint's nested `encoder/config.json`, `rl_agent_config.json`, and tokenizer
directory. Pin checkpoint revisions and record their provenance.

## 2. Establish parity — initial validation complete

See [Validation.md](Validation.md) for passing synthetic and trained-checkpoint
comparisons. Remaining work includes physical iOS validation, other checkpoints,
and representative routing evaluation. Custom boolean criteria now support
partial false/true overrides while preserving the default rubric and result API.
The `picodecisions` executable now provides completed
inference benchmarks and labeled candidate-routing evaluation; see
[Benchmarks.md](Benchmarks.md) for measurement scope and reproduction.

Use Swift Testing with reference fixtures for exact token IDs, prompt formatting,
option order, marker positions, attention masks, truncation, raw logits,
calibrated probabilities, selected answers, and action probabilities.

Preserve local/global RoPE settings, inclusive attention-window boundaries,
first-layer normalization behavior, and ReLU in the decision Transformer.
Resolve special-token IDs through the tokenizer. Test mixed question batches,
multiple languages, empty inputs, invalid options, capacity limits, and repeated
calls. Keep full-checkpoint tests opt-in and independent of network availability.

Benchmark completed inference, including tokenization and result processing.
Report cold loading separately. Establish FP16 parity before experimenting with
quantization or compilation. Author-reported Python performance is not a Swift
performance guarantee.

## 3. Tool-selection integration — adapter implemented

`ToolDecisionSelector` evaluates a bounded candidate set through `DecisionModel`
using tool descriptions and an explicit no-match choice. It preserves candidate
IDs, validates outputs, skips empty candidate sets, and keeps retrieval scores
as metadata separate from model probabilities. Callers own thresholds, argument
collection, authorization, execution, and fallback.

Laya reports original and retained token counts for state, instructions, and
every option. The selector and acceptance disposition preserve these diagnostics
so a consuming application can surface lost criteria. `.reject` rejects state
overflow while retaining upstream instruction/option budgets. See
[InputDiagnostics.md](InputDiagnostics.md).

`ToolDecisionAcceptancePolicy` optionally defers uncertain recommendations using
explicit selected-probability and runner-up-margin thresholds. It preserves raw
answers and keeps accepted no match separate from abstention. See
[ToolRouting.md](ToolRouting.md); application-specific threshold validation remains
outstanding.

Catalog retrieval remains in SmartToolSelection / PicoCore. Wiring the adapter into
a consuming application is still outstanding. The included handwritten routing
dataset exercises evaluation mechanics; its candidate ranks are supplied manually.

Evaluate retrieval-only and retrieval-plus-Laya against labeled requests,
including multiple-tool requests, negation, missing information, and no matching
tool. Measure candidate recall, final accuracy, false rejection, latency, and
memory before enabling candidate filtering. Preserve existing authorization and
fallback behavior in consuming applications.

## Development tooling — implemented

`Scripts/test.sh` provides core/CLI, synthetic inference, and opt-in real-checkpoint
validation. GitHub Actions runs core/CLI tests on hosted macOS; Metal tests require
a maintainer-dispatched physical runner. `Scripts/run.sh` builds a Release demo,
benchmark, and evaluator with Xcode. Robustness tests cover calls canceled before execution,
concurrent callers, malformed checkpoint files, and prompt capacity boundaries.

`Scripts/build-ios.sh` and the hosted iOS build matrix compile the library and
Metal resources for unsigned arm64 device/simulator destinations with an iOS 18
deployment target. Both local builds pass; physical-device inference remains
outstanding. See [IOSValidation.md](IOSValidation.md).

Hosted core/CLI CI has passed. Next: provision the Metal runner, collect a representative
held-out routing corpus, evaluate actual retrieval output, and assess calibration
or domain fine-tuning before enabling model-based candidate rejection.

## 4. Optional Core ML backend

Add `PicoDecisionsCoreML` with the same public decision contract. Ordinary exports
can reuse the provided Core ML graph. The ANE variant additionally needs host
embedding lookup, its specific tensor layout, and a CPU action head.

The published fast ANE bundle has a 96-token total budget, batch size one, and
32 option slots. Treat it as a specialized backend; evaluate larger exports
separately and validate on actual macOS/iOS deployment targets.

## Runtime boundaries

- Run entirely in-process; runtime Python and subprocesses are not required.
- Keep native model arrays inside their owning execution context.
- Share no private request or conversation state globally.
- Cache tokenized question prefixes where appropriate. Joint bidirectional
  encoding prevents arbitrary questions from sharing state hidden representations.
- Allow callers to supply local model directories and own model lifetimes.

Reference implementations reviewed for this plan:

- `mizorewww/laya-mlx` at `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`
- `mizorewww/laya-coreml` at `12b7501583c7f03a6b2e49ebe118a2c6302505b9`
- `PicoMLX/SmartToolSelection` at `d86079c05af1d6f799737b95415e68c8d8b551bc`
