import Foundation
import PicoDecisions

struct RoutingDataset: Decodable {
    let name: String
    let description: String
    let cases: [RoutingCase]

    func validate(maximumCandidates: Int) throws {
        guard !name.isEmpty, !description.isEmpty, !cases.isEmpty,
              Set(cases.map(\.id)).count == cases.count else {
            throw CLIError.usage("A dataset needs a name, description, nonempty cases, and unique case IDs.")
        }
        for item in cases {
            guard !item.id.isEmpty, item.candidates.count <= maximumCandidates,
                  Set(item.expectedToolIDs).count == item.expectedToolIDs.count,
                  item.expectedToolIDs.allSatisfy({
                      !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                      && $0 != ToolDecisionSelector.noMatchID
                  }) else {
                throw CLIError.usage("Invalid case ID, candidate count, or expected tool IDs in \(item.id).")
            }
            try DecisionRequest(state: item.state, questions: [.init(id: item.id,
                instructions: "Choose the matching tool.", kind: .choice(options:
                    item.candidates.map { .init(id: $0.id, description: $0.description) }
                    + [.init(id: ToolDecisionSelector.noMatchID, description: "No matching tool")]))]).validate()
            guard item.candidates.allSatisfy({
                !$0.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && $0.id != ToolDecisionSelector.noMatchID
                && !$0.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && ($0.retrievalScore?.isFinite ?? true)
            }) else { throw CLIError.usage("Invalid candidate in \(item.id).") }
            guard item.candidates.isEmpty || !item.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw CLIError.usage("A case with candidates needs a nonempty state: \(item.id).")
            }
        }
    }
}

struct RoutingCase: Decodable {
    struct Candidate: Decodable {
        let id: String
        let description: String
        let retrievalScore: Double?
        var candidate: ToolDecisionCandidate {
            .init(id: id, description: description, retrievalScore: retrievalScore)
        }
    }
    let id: String
    let state: String
    /// Any listed tool is an acceptable single selection; [] requires no match.
    let expectedToolIDs: [String]
    /// Supplied retrieval order; the first candidate is the retrieval-only baseline.
    let candidates: [Candidate]
}

struct RoutingOutcome: Encodable {
    let id: String
    let expectedToolIDs: [String]
    let candidateIDs: [String]
    let retrievalSelectedID: String?
    let selectedID: String?
    let candidateProbabilities: [PredictionReport.Probability]
    let noMatchProbability: Double?
    let latencyMilliseconds: Double
    let inputTokenCount: Int?
    let inputDiagnostics: DecisionInputDiagnostics?
    let confidence: Double?
    let actProbability: Double?
    /// Nil when no acceptance-policy flags were supplied; raw predictions remain above.
    let policy: RoutingPolicyOutcome?

    init(id: String, expectedToolIDs: [String], candidateIDs: [String], retrievalSelectedID: String?,
         selectedID: String?, candidateProbabilities: [PredictionReport.Probability],
         noMatchProbability: Double?, latencyMilliseconds: Double, inputTokenCount: Int?,
         confidence: Double?, actProbability: Double?, policy: RoutingPolicyOutcome? = nil,
         inputDiagnostics: DecisionInputDiagnostics? = nil) {
        self.id = id
        self.expectedToolIDs = expectedToolIDs
        self.candidateIDs = candidateIDs
        self.retrievalSelectedID = retrievalSelectedID
        self.selectedID = selectedID
        self.candidateProbabilities = candidateProbabilities
        self.noMatchProbability = noMatchProbability
        self.latencyMilliseconds = latencyMilliseconds
        self.inputTokenCount = inputTokenCount
        self.inputDiagnostics = inputDiagnostics
        self.confidence = confidence
        self.actProbability = actProbability
        self.policy = policy
    }

    func correct(_ selection: String?) -> Bool {
        if expectedToolIDs.isEmpty { return selection == nil }
        return selection.map { expectedToolIDs.contains($0) } ?? false
    }

    var availableExpectedIDs: Set<String> { Set(expectedToolIDs).intersection(candidateIDs) }

    func correctGivenCandidates(_ selection: String?) -> Bool {
        if availableExpectedIDs.isEmpty { return selection == nil }
        return selection.map { availableExpectedIDs.contains($0) } ?? false
    }
}

struct RoutingMetrics: Encodable {
    let caseCount: Int
    let matchingCaseCount: Int
    let noMatchCaseCount: Int
    let retrievalMissCaseCount: Int
    let selectableMatchingCaseCount: Int
    let retrievalAccuracy: Double
    let decisionAccuracy: Double
    let meanCandidateRecall: Double?
    let falseRejectionRate: Double?
    let falseAcceptanceRate: Double?
    let calibrationCaseCount: Int
    let expectedCalibrationError: Double?
    let inferredCaseCount: Int
    let inferredNoMatchCaseCount: Int
    let inferredFalseAcceptanceRate: Double?

    init(_ outcomes: [RoutingOutcome]) throws {
        guard !outcomes.isEmpty else { throw CLIError.usage("Cannot score an empty evaluation.") }
        caseCount = outcomes.count
        let matching = outcomes.filter { !$0.expectedToolIDs.isEmpty }
        let noMatch = outcomes.filter { $0.expectedToolIDs.isEmpty }
        matchingCaseCount = matching.count
        noMatchCaseCount = noMatch.count
        let selectable = matching.filter { !$0.availableExpectedIDs.isEmpty }
        selectableMatchingCaseCount = selectable.count
        retrievalMissCaseCount = matching.count - selectable.count
        retrievalAccuracy = Double(outcomes.filter { $0.correct($0.retrievalSelectedID) }.count) / Double(caseCount)
        decisionAccuracy = Double(outcomes.filter { $0.correct($0.selectedID) }.count) / Double(caseCount)
        meanCandidateRecall = matching.isEmpty ? nil : matching.reduce(0.0) { total, item in
            total + Double(Set(item.expectedToolIDs).intersection(item.candidateIDs).count) / Double(item.expectedToolIDs.count)
        } / Double(matching.count)
        falseRejectionRate = selectable.isEmpty ? nil
            : Double(selectable.filter { $0.selectedID == nil }.count) / Double(selectable.count)
        falseAcceptanceRate = noMatch.isEmpty ? nil
            : Double(noMatch.filter { $0.selectedID != nil }.count) / Double(noMatch.count)
        let inferred = outcomes.filter { !$0.candidateIDs.isEmpty }
        let inferredNoMatch = inferred.filter { $0.expectedToolIDs.isEmpty }
        inferredCaseCount = inferred.count
        inferredNoMatchCaseCount = inferredNoMatch.count
        inferredFalseAcceptanceRate = inferredNoMatch.isEmpty ? nil
            : Double(inferredNoMatch.filter { $0.selectedID != nil }.count) / Double(inferredNoMatch.count)

        // Calibration uses the selected answer probability, not entropy confidence.
        // Empty candidate sets have no model distribution and are excluded.
        let calibration = outcomes.compactMap { item -> (Double, Double)? in
            guard let noMatch = item.noMatchProbability else { return nil }
            let probability = item.selectedID.flatMap { selected in
                item.candidateProbabilities.first { $0.id == selected }?.probability
            } ?? noMatch
            return (probability, item.correctGivenCandidates(item.selectedID) ? 1 : 0)
        }
        calibrationCaseCount = calibration.count
        if calibration.isEmpty { expectedCalibrationError = nil }
        else {
            var error = 0.0
            for bin in 0..<10 {
                let values = calibration.filter { min(9, Int($0.0 * 10)) == bin }
                if values.isEmpty { continue }
                let confidence = values.reduce(0.0) { $0 + $1.0 } / Double(values.count)
                let accuracy = values.reduce(0.0) { $0 + $1.1 } / Double(values.count)
                error += Double(values.count) / Double(calibration.count) * abs(confidence - accuracy)
            }
            expectedCalibrationError = error
        }
    }
}

struct RoutingPolicyConfiguration: Encodable {
    let minimumProbability: Double
    let minimumMargin: Double
    let minimumProbabilitySource: String
    let minimumMarginSource: String

    init?(minimumProbability: Double?, minimumMargin: Double?) {
        guard minimumProbability != nil || minimumMargin != nil else { return nil }
        self.minimumProbability = minimumProbability ?? 0
        self.minimumMargin = minimumMargin ?? 0
        minimumProbabilitySource = minimumProbability == nil ? "defaultZero" : "commandLine"
        minimumMarginSource = minimumMargin == nil ? "defaultZero" : "commandLine"
    }
}

struct RoutingPolicyOutcome: Encodable {
    let status: String
    let reasons: [String]
    let selectedProbability: Double?
    let probabilityMargin: Double?

    init(_ disposition: ToolDecisionDisposition) {
        status = disposition.status.rawValue
        reasons = disposition.reasons.map(\.rawValue)
        selectedProbability = disposition.selectedProbability
        probabilityMargin = disposition.probabilityMargin
    }

    var accepted: Bool { status == "acceptedTool" || status == "acceptedNoMatch" }
}

/// Every denominator excludes deterministic empty-candidate results.
struct RoutingPolicyMetrics: Encodable {
    let inferredCaseCount: Int
    let acceptedCaseCount: Int
    let acceptedToolCaseCount: Int
    let acceptedNoMatchCaseCount: Int
    let abstainedCaseCount: Int
    let coverage: Double?
    let abstentionRate: Double?
    let decisionAccuracy: Double?
    let candidateRelativeAccuracy: Double?
    let acceptedAccuracy: Double?
    let acceptedCandidateRelativeAccuracy: Double?
    let inferredNoMatchCaseCount: Int
    let falseAcceptanceRate: Double?

    init(_ outcomes: [RoutingOutcome]) {
        let inferred = outcomes.filter { !$0.candidateIDs.isEmpty }
        let accepted = inferred.filter { $0.policy?.accepted == true }
        let abstained = inferred.filter { $0.policy?.status == "abstained" }
        inferredCaseCount = inferred.count
        acceptedCaseCount = accepted.count
        acceptedToolCaseCount = accepted.filter { $0.policy?.status == "acceptedTool" }.count
        acceptedNoMatchCaseCount = accepted.filter { $0.policy?.status == "acceptedNoMatch" }.count
        abstainedCaseCount = abstained.count
        let correct = accepted.filter { $0.correct($0.selectedID) }.count
        let candidateCorrect = accepted.filter { $0.correctGivenCandidates($0.selectedID) }.count
        coverage = Self.rate(accepted.count, inferred.count)
        abstentionRate = Self.rate(abstained.count, inferred.count)
        // An abstention is never credited as a correct no match.
        decisionAccuracy = Self.rate(correct, inferred.count)
        candidateRelativeAccuracy = Self.rate(candidateCorrect, inferred.count)
        acceptedAccuracy = Self.rate(correct, accepted.count)
        acceptedCandidateRelativeAccuracy = Self.rate(candidateCorrect, accepted.count)
        let noMatch = inferred.filter { $0.expectedToolIDs.isEmpty }
        inferredNoMatchCaseCount = noMatch.count
        falseAcceptanceRate = Self.rate(noMatch.filter { $0.policy?.status == "acceptedTool" }.count, noMatch.count)
    }

    private static func rate(_ count: Int, _ denominator: Int) -> Double? {
        denominator == 0 ? nil : Double(count) / Double(denominator)
    }
}

struct EvaluationReport: Encodable {
    let schemaVersion = 2
    let timestamp = Date()
    let runtime: RuntimeMetadata
    let checkpoint: CheckpointMetadata
    let precision: String
    let batchSize: Int
    let datasetName: String
    let datasetDescription: String
    let datasetSHA256: String
    let loadMilliseconds: Double
    let maximumCandidates: Int
    let metrics: RoutingMetrics
    let policyConfiguration: RoutingPolicyConfiguration?
    let policyMetrics: RoutingPolicyMetrics?
    let latency: LatencyStatistics
    let memory: MLXMemoryReport
    let outcomes: [RoutingOutcome]
}

// Keep the evaluation schema independent of MLX's own Codable layout.
struct MLXMemoryReport: Encodable {
    let activeBytes: Int
    let cacheBytes: Int
    let peakActiveBytes: Int
}
