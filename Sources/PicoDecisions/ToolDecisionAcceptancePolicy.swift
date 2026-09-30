import Foundation

/// A reason a caller's configured policy declined the raw recommendation.
public enum ToolDecisionAbstentionReason: String, Sendable, Equatable {
    case belowMinimumProbability
    case belowMinimumMargin
    /// The supplied selection did not contain a valid complete distribution.
    case invalidSelection
}

/// The outcome of applying an explicit acceptance policy after inference.
///
/// The original selection is retained, including when the policy abstains. An
/// abstention is an uncertain recommendation, distinct from a no-match answer.
public struct ToolDecisionDisposition: Sendable, Equatable {
    public enum Status: String, Sendable, Equatable {
        case acceptedTool
        case acceptedNoMatch
        case abstained
    }

    public let selection: ToolDecisionSelection
    public let status: Status
    public let reasons: [ToolDecisionAbstentionReason]
    /// The raw probability of the selected tool or selected no-match answer.
    /// Nil when inference was skipped or the selection was malformed.
    public let selectedProbability: Double?
    /// Selected probability minus the strongest rival, including no match.
    /// Negative values are retained. Nil when inference was skipped or invalid.
    public let probabilityMargin: Double?
}

/// An optional caller-owned policy applied to an existing tool recommendation.
///
/// Both thresholds are explicit. Zero disables its gate; zero for both preserves
/// every valid raw recommendation. Positive thresholds apply equally to tool
/// and no-match answers, and equality passes. Choose thresholds using held-out
/// application data. Applying this policy does not rerun or improve the model,
/// authorize a tool, collect arguments, or execute an action.
public struct ToolDecisionAcceptancePolicy: Sendable, Equatable {
    public let minimumProbability: Double
    public let minimumMargin: Double

    public init(minimumProbability: Double, minimumMargin: Double) throws {
        guard minimumProbability.isFinite, minimumMargin.isFinite,
              (0...1).contains(minimumProbability), (0...1).contains(minimumMargin) else {
            throw DecisionError.invalidConfiguration("Tool acceptance thresholds must be finite values between zero and one.")
        }
        self.minimumProbability = minimumProbability
        self.minimumMargin = minimumMargin
    }

    public func evaluate(_ selection: ToolDecisionSelection) -> ToolDecisionDisposition {
        func disposition(_ status: ToolDecisionDisposition.Status,
                         reasons: [ToolDecisionAbstentionReason] = [],
                         probability: Double? = nil, margin: Double? = nil) -> ToolDecisionDisposition {
            .init(selection: selection, status: status, reasons: reasons,
                  selectedProbability: probability, probabilityMargin: margin)
        }
        func invalid() -> ToolDecisionDisposition {
            disposition(.abstained, reasons: [.invalidSelection])
        }

        // Empty candidate input has no model distribution. Preserve this
        // deterministic no-match without manufacturing a probability of one.
        if selection.candidateProbabilities.isEmpty {
            guard selection.selectedCandidate == nil, selection.noMatchProbability == nil,
                  selection.confidence == nil, selection.actProbability == nil,
                  selection.inputTokenCount == nil, selection.inputDiagnostics == nil else { return invalid() }
            return disposition(.acceptedNoMatch)
        }

        // ToolDecisionSelector already validates its output. Defensively refuse
        // incomplete or malformed selections if other in-module code supplies one.
        guard let noMatch = selection.noMatchProbability, noMatch.isFinite,
              (0...1).contains(noMatch), selection.confidence?.isFinite != false,
              selection.actProbability?.isFinite != false,
              selection.actProbability.map({ (0...1).contains($0) }) != false,
              selection.inputTokenCount.map({ $0 >= 0 }) != false else { return invalid() }
        var probabilities: [String: Double] = [ToolDecisionSelector.noMatchID: noMatch]
        for item in selection.candidateProbabilities {
            guard !item.optionID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  probabilities[item.optionID] == nil, item.probability.isFinite,
                  (0...1).contains(item.probability) else { return invalid() }
            probabilities[item.optionID] = item.probability
        }
        guard abs(probabilities.values.reduce(0, +) - 1) <= 1e-5 else { return invalid() }
        if let candidate = selection.selectedCandidate {
            guard candidate.id != ToolDecisionSelector.noMatchID,
                  !candidate.description.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  candidate.retrievalScore?.isFinite != false else { return invalid() }
        }
        let selectedID = selection.selectedCandidateID ?? ToolDecisionSelector.noMatchID
        guard let probability = probabilities[selectedID],
              let rival = probabilities.filter({ $0.key != selectedID }).map(\.value).max() else {
            return invalid()
        }
        let margin = probability - rival
        var reasons: [ToolDecisionAbstentionReason] = []
        if minimumProbability > 0, probability < minimumProbability {
            reasons.append(.belowMinimumProbability)
        }
        if minimumMargin > 0, margin < minimumMargin {
            reasons.append(.belowMinimumMargin)
        }
        let status: ToolDecisionDisposition.Status = reasons.isEmpty
            ? (selection.selectedCandidate == nil ? .acceptedNoMatch : .acceptedTool) : .abstained
        return disposition(status, reasons: reasons, probability: probability, margin: margin)
    }
}
