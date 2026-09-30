# Inspect prompt truncation

`LayaModel` reports `DecisionResult.inputDiagnostics` after every inferred
question. `ToolDecisionSelector` preserves the same diagnostics in
`ToolDecisionSelection.inputDiagnostics`, and an acceptance disposition retains
them in its original selection.

Inspect these counts when testing long requests or tool descriptions:

```swift
let selection = try await selector.select(query: request, candidates: retrievedTools)
if let diagnostics = selection.inputDiagnostics {
    print("Prompt shortened:", diagnostics.wasTruncated)
    print("State tokens:", diagnostics.state.originalTokenCount,
          "retained:", diagnostics.state.retainedTokenCount)
    print("Instruction tokens:", diagnostics.instructions.originalTokenCount,
          "retained:", diagnostics.instructions.retainedTokenCount)
    for option in diagnostics.options where option.tokens.wasTruncated {
        print("Option shortened:", option.optionID,
              option.tokens.originalTokenCount, "to", option.tokens.retainedTokenCount)
    }
}
```

The example assumes an initialized selector, a request string, and retrieved
`ToolDecisionCandidate` values. Empty candidate input skips inference and has nil
diagnostics. For an injected backend, nil means diagnostics are unavailable;
it does not establish that all text was retained. Diagnostics do not change the
prediction or acceptance policy. The application decides whether to shorten
text, retry, or defer a recommendation whose criteria were truncated.

## Interpret the counts

`DecisionInputDiagnostics` contains `state`, `instructions`, and ordered
`options`. Each component's `TokenCounts` has `originalTokenCount`,
`retainedTokenCount`, and a computed `wasTruncated`. The top-level
`wasTruncated` is true if any component lost tokens.

Counts describe the cleaned, formatted text actually used by the tokenizer.
Laya replaces literal mask-token text with spaces before tokenization. The
instruction count includes the question type prefix, and option counts include
the option label and leading space. Structural CLS, SEP, and MASK tokens are
excluded, so the sum of retained component counts is smaller than
`inputTokenCount`.

| Option kind | Diagnostic `optionID` |
| --- | --- |
| Choice | The original `DecisionOption.id`, including the selector's no-match ID |
| Score | Zero-based level index as a decimal string, such as `"0"` |
| Boolean | `"false"` followed by `"true"` |

Each question has its own remaining state budget. A request with several
questions can therefore retain different amounts of the same state.

## Understand Laya's limits

`inputPolicy: .reject` rejects oversized **state** only. Instructions and options
still follow the upstream prefix limits under both input policies:

- Each formatted option initially retains at most 48 text tokens, plus its MASK
  marker.
- If options leave less than 16 tokens in the configured head budget, the model
  shortens each option further. Diagnostics report the final retained count.
- Instructions retain the prefix allowed by the remaining head budget, with an
  upstream minimum budget of 8 tokens.
- `.truncateState` retains the beginning of state that fits each question's
  remaining context. `.reject` throws `DecisionError.capacityExceeded` instead.

No option is dropped to make a prompt fit. A question that cannot fit even with
these limits throws a capacity error and produces no result diagnostics.

## Inspect CLI reports

`demo` predictions and `evaluate` outcomes include optional `inputDiagnostics`.
Benchmark workloads include an ordered `inputDiagnostics` array for the first
prediction when every question reports diagnostics. Existing raw predictions,
metrics, and token counts remain unchanged. Nil diagnostics are omitted from
JSON. Encoded diagnostics include the aggregate `wasTruncated` and original and
retained counts for each component.
