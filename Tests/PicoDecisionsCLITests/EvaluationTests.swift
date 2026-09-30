import Foundation
import PicoDecisions
import Testing
@testable import PicoDecisionsCLI

@Test func latencyUsesMedianAndNearestRankP95() throws {
    let latency = try LatencyStatistics([20, 1, 2, 3])
    #expect(latency.medianMilliseconds == 2.5)
    #expect(latency.p95Milliseconds == 20)
    #expect(latency.meanMilliseconds == 6.5)
    #expect(throws: CLIError.self) { try LatencyStatistics([]) }
    #expect(throws: CLIError.self) { try LatencyStatistics([.nan]) }
}

@Test func optionalInputDiagnosticsJSONPreservesRawPredictionFields() throws {
    let diagnostics = DecisionInputDiagnostics(
        state: .init(originalTokenCount: 100, retainedTokenCount: 20),
        instructions: .init(originalTokenCount: 8, retainedTokenCount: 8),
        options: [.init(optionID: "weather", tokens: .init(originalTokenCount: 60, retainedTokenCount: 48))])
    func result(_ diagnostics: DecisionInputDiagnostics?) -> DecisionResult {
        .init(id: "route", answer: .choice(selectedID: "weather", probabilities: [
            .init(optionID: "weather", probability: 1)]), inputTokenCount: 81, inputDiagnostics: diagnostics)
    }
    let encoder = JSONEncoder()
    let without = try #require(JSONSerialization.jsonObject(with: encoder.encode(
        PredictionReport(result(nil)))) as? [String: Any])
    var with = try #require(JSONSerialization.jsonObject(with: encoder.encode(
        PredictionReport(result(diagnostics)))) as? [String: Any])
    #expect(without["inputDiagnostics"] == nil)
    let encodedDiagnostics = try #require(with.removeValue(forKey: "inputDiagnostics") as? [String: Any])
    #expect(encodedDiagnostics["wasTruncated"] as? Bool == true)
    let state = try #require(encodedDiagnostics["state"] as? [String: Any])
    #expect(state["originalTokenCount"] as? Int == 100)
    #expect(state["retainedTokenCount"] as? Int == 20)
    let options = try #require(encodedDiagnostics["options"] as? [[String: Any]])
    #expect(options.first?["optionID"] as? String == "weather")
    #expect(NSDictionary(dictionary: with).isEqual(to: without))
    #expect(try JSONDecoder().decode(DecisionInputDiagnostics.self, from: encoder.encode(diagnostics)) == diagnostics)

    let outcome = RoutingOutcome(id: "route", expectedToolIDs: ["weather"], candidateIDs: ["weather"],
        retrievalSelectedID: "weather", selectedID: "weather", candidateProbabilities: [.init(id: "weather", probability: 1)],
        noMatchProbability: 0, latencyMilliseconds: 1, inputTokenCount: 81, confidence: nil, actProbability: nil,
        inputDiagnostics: diagnostics)
    let routingJSON = try #require(JSONSerialization.jsonObject(with: encoder.encode(outcome)) as? [String: Any])
    let routingDiagnostics = try #require(routingJSON["inputDiagnostics"] as? [String: Any])
    #expect(NSDictionary(dictionary: routingDiagnostics).isEqual(to: encodedDiagnostics))
}

@Test func separatesRetrievalMissesRejectionAndNoMatch() throws {
    func outcome(_ id: String, expected: [String], candidates: [String], selected: String?) -> RoutingOutcome {
        .init(id: id, expectedToolIDs: expected, candidateIDs: candidates,
              retrievalSelectedID: candidates.first, selectedID: selected,
              candidateProbabilities: [], noMatchProbability: nil,
              latencyMilliseconds: 1, inputTokenCount: nil, confidence: nil, actProbability: nil)
    }
    let metrics = try RoutingMetrics([
        outcome("reranked", expected: ["a"], candidates: ["b", "a"], selected: "a"),
        outcome("retrieval-miss", expected: ["a"], candidates: ["b"], selected: nil),
        outcome("false-accept", expected: [], candidates: ["b"], selected: "b"),
        outcome("empty", expected: [], candidates: [], selected: nil),
        outcome("multiple", expected: ["a", "b"], candidates: ["b"], selected: "b")
    ])
    #expect(metrics.retrievalAccuracy == 0.4)
    #expect(metrics.decisionAccuracy == 0.6)
    #expect(metrics.meanCandidateRecall == 0.5)
    #expect(metrics.retrievalMissCaseCount == 1)
    #expect(metrics.selectableMatchingCaseCount == 2)
    #expect(metrics.falseRejectionRate == 0)
    #expect(metrics.falseAcceptanceRate == 0.5)
    #expect(metrics.inferredCaseCount == 4)
    #expect(metrics.inferredNoMatchCaseCount == 1)
    #expect(metrics.inferredFalseAcceptanceRate == 1)
    #expect(metrics.expectedCalibrationError == nil)
}

@Test func calibrationTreatsNoMatchAsCorrectWhenRetrievalMisses() throws {
    let metrics = try RoutingMetrics([.init(id: "missing", expectedToolIDs: ["a"], candidateIDs: ["b"],
        retrievalSelectedID: "b", selectedID: nil, candidateProbabilities: [.init(id: "b", probability: 0)],
        noMatchProbability: 1, latencyMilliseconds: 1, inputTokenCount: nil, confidence: nil, actProbability: nil)])
    #expect(metrics.decisionAccuracy == 0)
    #expect(metrics.retrievalMissCaseCount == 1)
    #expect(metrics.falseRejectionRate == nil)
    #expect(metrics.expectedCalibrationError == 0)
}

@Test func rejectsReportPathsThatOverwriteInputs() {
    for arguments in [
        ["evaluate", "--model", "tiny", "--dataset", "cases.json", "--output", "cases.json"],
        ["demo", "--model", "tiny", "--output", "tiny/model.safetensors"],
        ["benchmark", "--model", "tiny", "--state-file", "input.txt", "--output", "input.txt"]
    ] { #expect(throws: CLIError.self) { try Options(arguments) } }
}

@Test func protectsCheckpointFilesSymlinkedOutsideModelDirectory() throws {
    let directory = FileManager.default.temporaryDirectory.appending(path: "PicoDecisions-output-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: directory) }
    let model = directory.appending(path: "model")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    let weights = directory.appending(path: "cached-weights.safetensors")
    try Data([1, 2, 3]).write(to: weights)
    try FileManager.default.createSymbolicLink(at: model.appending(path: "model.safetensors"), withDestinationURL: weights)
    #expect(throws: CLIError.self) {
        try Options(["demo", "--model", model.path, "--output", weights.path])
    }
}

@Test func calibrationUsesSelectedProbabilityAndIncludesProbabilityOne() throws {
    let outcomes = [RoutingOutcome(id: "wrong", expectedToolIDs: ["a"], candidateIDs: ["a", "b"],
        retrievalSelectedID: "a", selectedID: "b",
        candidateProbabilities: [.init(id: "a", probability: 0), .init(id: "b", probability: 1)],
        noMatchProbability: 0, latencyMilliseconds: 1, inputTokenCount: nil,
        confidence: 0.1, actProbability: nil)]
    let metrics = try RoutingMetrics(outcomes)
    #expect(metrics.calibrationCaseCount == 1)
    #expect(metrics.expectedCalibrationError == 1)
}

@Test func refusesMalformedAndUnrelatedCommandFlags() throws {
    let invalid = [
        ["demo", "--model", "tiny", "--iterations", "1"],
        ["benchmark", "--model", "tiny", "--iterations", "0"],
        ["benchmark", "--model", "tiny", "--questions", "1,,8"],
        ["benchmark", "--model", "tiny", "--questions", "1,1"],
        ["demo", "--model", "tiny", "--model", "other"],
        ["evaluate", "--model", "tiny"]
    ]
    for arguments in invalid { #expect(throws: CLIError.self) { try Options(arguments) } }
    let options = try Options(["benchmark", "--model", "tiny", "--questions", "1,8", "--warmup", "0"])
    #expect(options.questionCounts == [1, 8])
    #expect(options.warmup == 0)
}

@Test func datasetAllowsMissingCandidatesButRejectsInvalidLabels() throws {
    let valid = RoutingDataset(name: "test", description: "test", cases: [.init(id: "missing",
        state: "Find it", expectedToolIDs: ["missing-tool"], candidates: [])])
    try valid.validate(maximumCandidates: 8)
    for expectedToolIDs in [[], [" weather "]] {
        let dataset = RoutingDataset(name: "test", description: "test", cases: [.init(id: "valid",
            state: "Check the weather", expectedToolIDs: expectedToolIDs,
            candidates: [.init(id: " weather ", description: "Get the weather", retrievalScore: nil)])])
        try dataset.validate(maximumCandidates: 8)
    }
    let invalid = RoutingDataset(name: "test", description: "test", cases: [.init(id: "bad",
        state: "Find it", expectedToolIDs: ["__pico_no_match__"], candidates: [])])
    #expect(throws: CLIError.self) { try invalid.validate(maximumCandidates: 8) }
}

@Test(arguments: [[""], [" "], ["\t"], ["\n"], [" \t\r\n "], ["\u{00A0}"], ["weather", " "]])
func datasetRejectsBlankExpectedToolIDs(expectedToolIDs: [String]) throws {
    let dataset = RoutingDataset(name: "test", description: "test", cases: [.init(id: "bad-label",
        state: "Check the weather", expectedToolIDs: expectedToolIDs,
        candidates: [.init(id: "weather", description: "Get the weather", retrievalScore: nil)])])
    #expect(throws: CLIError.self) { try dataset.validate(maximumCandidates: 8) }
}

@Test func policyFlagsAreOptionalAndRecordEffectiveThresholdProvenance() throws {
    let base = ["evaluate", "--model", "tiny", "--dataset", "cases.json"]
    let disabled = try Options(base)
    #expect(disabled.minimumProbability == nil)
    #expect(disabled.minimumMargin == nil)
    #expect(RoutingPolicyConfiguration(minimumProbability: disabled.minimumProbability,
                                       minimumMargin: disabled.minimumMargin) == nil)

    let probabilityOnly = try Options(base + ["--minimum-probability", "0.8"])
    let probabilityConfiguration = try #require(RoutingPolicyConfiguration(
        minimumProbability: probabilityOnly.minimumProbability, minimumMargin: probabilityOnly.minimumMargin))
    #expect(probabilityConfiguration.minimumProbability == 0.8)
    #expect(probabilityConfiguration.minimumMargin == 0)
    #expect(probabilityConfiguration.minimumProbabilitySource == "commandLine")
    #expect(probabilityConfiguration.minimumMarginSource == "defaultZero")

    let marginOnly = try Options(base + ["--minimum-margin", "0.25"])
    let marginConfiguration = try #require(RoutingPolicyConfiguration(
        minimumProbability: marginOnly.minimumProbability, minimumMargin: marginOnly.minimumMargin))
    #expect(marginConfiguration.minimumProbability == 0)
    #expect(marginConfiguration.minimumMargin == 0.25)
    #expect(marginConfiguration.minimumProbabilitySource == "defaultZero")
    #expect(marginConfiguration.minimumMarginSource == "commandLine")

    for value in ["0", "1"] {
        let boundaries = try Options(base + ["--minimum-probability", value, "--minimum-margin", value])
        #expect(boundaries.minimumProbability == Double(value))
        #expect(boundaries.minimumMargin == Double(value))
    }
}

@Test(arguments: ["nan", "NaN", "inf", "-inf", "infinity", "1e309", "-0.01", "1.01", "text", ""])
func rejectsInvalidPolicyThresholds(value: String) {
    for flag in ["--minimum-probability", "--minimum-margin"] {
        #expect(throws: CLIError.self) {
            try Options(["evaluate", "--model", "tiny", "--dataset", "cases.json", flag, value])
        }
    }
}

@Test func rejectsDuplicateIncompleteAndUnrelatedPolicyFlags() {
    for flag in ["--minimum-probability", "--minimum-margin"] {
        let invalid = [
            ["evaluate", "--model", "tiny", "--dataset", "cases.json", flag, "0.5", flag, "0.6"],
            ["evaluate", "--model", "tiny", "--dataset", "cases.json", flag],
            ["evaluate", "--model", "tiny", flag, "0.5"],
            ["demo", "--model", "tiny", flag, "0.5"],
            ["benchmark", "--model", "tiny", flag, "0.5"]
        ]
        for arguments in invalid { #expect(throws: CLIError.self) { try Options(arguments) } }
    }
}

@Test func inferredFalseAcceptanceExcludesDeterministicEmptyCandidates() throws {
    let outcomes = [
        RoutingOutcome(id: "wrong-tool", expectedToolIDs: [], candidateIDs: ["b"], retrievalSelectedID: "b",
            selectedID: "b", candidateProbabilities: [.init(id: "b", probability: 0.9)], noMatchProbability: 0.1,
            latencyMilliseconds: 1, inputTokenCount: 5, confidence: nil, actProbability: nil),
        RoutingOutcome(id: "no-match", expectedToolIDs: [], candidateIDs: ["b"], retrievalSelectedID: "b",
            selectedID: nil, candidateProbabilities: [.init(id: "b", probability: 0.1)], noMatchProbability: 0.9,
            latencyMilliseconds: 1, inputTokenCount: 5, confidence: nil, actProbability: nil),
        RoutingOutcome(id: "empty", expectedToolIDs: [], candidateIDs: [], retrievalSelectedID: nil,
            selectedID: nil, candidateProbabilities: [], noMatchProbability: nil,
            latencyMilliseconds: 0, inputTokenCount: nil, confidence: nil, actProbability: nil)
    ]
    let metrics = try RoutingMetrics(outcomes)
    #expect(metrics.noMatchCaseCount == 3)
    #expect(metrics.falseAcceptanceRate == 1.0 / 3)
    #expect(metrics.inferredCaseCount == 2)
    #expect(metrics.inferredNoMatchCaseCount == 2)
    #expect(metrics.inferredFalseAcceptanceRate == 0.5)
}

@Test func policyMetricsSeparateAbstentionFromAcceptedNoMatch() async throws {
    let policy = try ToolDecisionAcceptancePolicy(minimumProbability: 0.8, minimumMargin: 0.1)
    let outcomes = try await [
        evaluationOutcome("correct-tool", expected: ["a"], candidates: ["a"], selected: "a", toolProbability: 0.9, policy: policy),
        evaluationOutcome("correct-no-match", expected: [], candidates: ["b"], selected: nil, toolProbability: 0.1, policy: policy),
        evaluationOutcome("retrieval-miss", expected: ["a"], candidates: ["b"], selected: nil, toolProbability: 0.1, policy: policy),
        evaluationOutcome("abstain-match", expected: ["a"], candidates: ["a"], selected: "a", toolProbability: 0.6, policy: policy),
        evaluationOutcome("abstain-no-match", expected: [], candidates: ["b"], selected: nil, toolProbability: 0.4, policy: policy),
        evaluationOutcome("false-accept", expected: [], candidates: ["b"], selected: "b", toolProbability: 0.95, policy: policy),
        evaluationOutcome("empty-no-match", expected: [], candidates: [], selected: nil, toolProbability: 0, policy: policy),
        evaluationOutcome("empty-missing-tool", expected: ["a"], candidates: [], selected: nil, toolProbability: 0, policy: policy)
    ]
    let metrics = RoutingPolicyMetrics(outcomes)
    #expect(metrics.inferredCaseCount == 6)
    #expect(metrics.acceptedCaseCount == 4)
    #expect(metrics.acceptedToolCaseCount == 2)
    #expect(metrics.acceptedNoMatchCaseCount == 2)
    #expect(metrics.abstainedCaseCount == 2)
    #expect(metrics.coverage == 2.0 / 3)
    #expect(metrics.abstentionRate == 1.0 / 3)
    #expect(metrics.decisionAccuracy == 1.0 / 3)
    #expect(metrics.candidateRelativeAccuracy == 0.5)
    #expect(metrics.acceptedAccuracy == 0.5)
    #expect(metrics.acceptedCandidateRelativeAccuracy == 0.75)
    #expect(metrics.inferredNoMatchCaseCount == 3)
    #expect(metrics.falseAcceptanceRate == 1.0 / 3)
    #expect(outcomes[1].selectedID == nil)
    #expect(outcomes[1].policy?.status == "acceptedNoMatch")
    #expect(outcomes[4].selectedID == nil)
    #expect(outcomes[4].policy?.status == "abstained")
    #expect(outcomes[4].policy?.reasons == ["belowMinimumProbability"])
}

@Test func policyMetricsHandleNoInferenceAndNoAcceptedCases() async throws {
    let policy = try ToolDecisionAcceptancePolicy(minimumProbability: 1, minimumMargin: 1)
    let empty = try await evaluationOutcome("empty", expected: [], candidates: [], selected: nil,
                                             toolProbability: 0, policy: policy)
    let emptyMetrics = RoutingPolicyMetrics([empty])
    #expect(emptyMetrics.inferredCaseCount == 0)
    #expect(emptyMetrics.acceptedCaseCount == 0)
    #expect(emptyMetrics.acceptedNoMatchCaseCount == 0)
    #expect(emptyMetrics.coverage == nil)
    #expect(emptyMetrics.abstentionRate == nil)
    #expect(emptyMetrics.acceptedAccuracy == nil)
    #expect(emptyMetrics.candidateRelativeAccuracy == nil)
    #expect(emptyMetrics.falseAcceptanceRate == nil)

    let abstained = try await evaluationOutcome("abstained", expected: [], candidates: ["b"], selected: nil,
                                                toolProbability: 0.4, policy: policy)
    let metrics = RoutingPolicyMetrics([empty, abstained])
    #expect(metrics.inferredCaseCount == 1)
    #expect(metrics.acceptedCaseCount == 0)
    #expect(metrics.abstainedCaseCount == 1)
    #expect(metrics.coverage == 0)
    #expect(metrics.abstentionRate == 1)
    #expect(metrics.decisionAccuracy == 0)
    #expect(metrics.candidateRelativeAccuracy == 0)
    #expect(metrics.acceptedAccuracy == nil)
    #expect(metrics.acceptedCandidateRelativeAccuracy == nil)
    #expect(metrics.falseAcceptanceRate == 0)
    #expect(throws: CLIError.self) { try RoutingMetrics([]) }
}

@Test func optionalPolicyJSONPreservesRawOutcomesAndMetrics() async throws {
    let policy = try ToolDecisionAcceptancePolicy(minimumProbability: 0.8, minimumMargin: 0.1)
    let disabled = try await evaluationOutcome("same", expected: [], candidates: ["b"], selected: nil,
                                               toolProbability: 0.4, policy: nil)
    let enabled = try await evaluationOutcome("same", expected: [], candidates: ["b"], selected: nil,
                                              toolProbability: 0.4, policy: policy)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    var enabledJSON = try #require(JSONSerialization.jsonObject(with: encoder.encode(enabled)) as? [String: Any])
    let disabledJSON = try #require(JSONSerialization.jsonObject(with: encoder.encode(disabled)) as? [String: Any])
    #expect(disabledJSON["policy"] == nil)
    let removedPolicy = enabledJSON.removeValue(forKey: "policy")
    let policyJSON = try #require(removedPolicy as? [String: Any])
    #expect(policyJSON["status"] as? String == "abstained")
    #expect(policyJSON["reasons"] as? [String] == ["belowMinimumProbability"])
    #expect(policyJSON["selectedProbability"] as? Double == 0.6)
    #expect(abs(try #require(policyJSON["probabilityMargin"] as? Double) - 0.2) < 1e-12)
    #expect(NSDictionary(dictionary: enabledJSON).isEqual(to: disabledJSON))
    #expect(try encoder.encode(RoutingMetrics([enabled])) == encoder.encode(RoutingMetrics([disabled])))

    let configuration = try #require(RoutingPolicyConfiguration(minimumProbability: nil, minimumMargin: 0.25))
    let configurationJSON = try #require(JSONSerialization.jsonObject(with: encoder.encode(configuration)) as? [String: Any])
    #expect(configurationJSON["minimumProbability"] as? Double == 0)
    #expect(configurationJSON["minimumProbabilitySource"] as? String == "defaultZero")
    #expect(configurationJSON["minimumMargin"] as? Double == 0.25)
    #expect(configurationJSON["minimumMarginSource"] as? String == "commandLine")
}

private struct EvaluationFixedModel: DecisionModel {
    let selectedID: String
    let probabilities: [OptionProbability]

    func predict(_ request: DecisionRequest) async throws -> [DecisionResult] {
        [.init(id: ToolDecisionSelector.questionID, answer: .choice(selectedID: selectedID, probabilities: probabilities),
               confidence: 0.123, actProbability: 0.9, inputTokenCount: 5)]
    }
}

private func evaluationOutcome(_ id: String, expected: [String], candidates: [String], selected: String?,
                               toolProbability: Double, policy: ToolDecisionAcceptancePolicy?) async throws -> RoutingOutcome {
    let probabilities = candidates.map { OptionProbability(optionID: $0, probability: toolProbability) }
        + [OptionProbability(optionID: ToolDecisionSelector.noMatchID, probability: 1 - toolProbability)]
    let selector = try ToolDecisionSelector(model: EvaluationFixedModel(
        selectedID: selected ?? ToolDecisionSelector.noMatchID, probabilities: probabilities))
    let selection = try await selector.select(query: "Choose a tool", candidates: candidates.map {
        .init(id: $0, description: "Tool \($0)")
    })
    return .init(id: id, expectedToolIDs: expected, candidateIDs: candidates, retrievalSelectedID: candidates.first,
        selectedID: selection.selectedCandidateID,
        candidateProbabilities: selection.candidateProbabilities.map { .init(id: $0.optionID, probability: $0.probability) },
        noMatchProbability: selection.noMatchProbability, latencyMilliseconds: 1,
        inputTokenCount: selection.inputTokenCount, confidence: selection.confidence, actProbability: selection.actProbability,
        policy: policy.map { RoutingPolicyOutcome($0.evaluate(selection)) })
}
