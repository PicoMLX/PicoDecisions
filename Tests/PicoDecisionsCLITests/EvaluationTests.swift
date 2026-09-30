import Foundation
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
    let invalid = RoutingDataset(name: "test", description: "test", cases: [.init(id: "bad",
        state: "Find it", expectedToolIDs: ["__pico_no_match__"], candidates: [])])
    #expect(throws: CLIError.self) { try invalid.validate(maximumCandidates: 8) }
}
