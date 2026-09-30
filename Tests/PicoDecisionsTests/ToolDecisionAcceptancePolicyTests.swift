import Testing
@testable import PicoDecisions

private func acceptanceSelection(selectedID: String? = "a", probabilityA: Double = 0.75,
                                 probabilityB: Double = 0.125, noMatch: Double = 0.125,
                                 confidence: Double? = 0.2, actProbability: Double? = 0.1) -> ToolDecisionSelection {
    ToolDecisionSelection(
        selectedCandidate: selectedID.map { .init(id: $0, description: "Tool \($0)", retrievalScore: -4) },
        candidateProbabilities: [.init(optionID: "a", probability: probabilityA),
                                 .init(optionID: "b", probability: probabilityB)],
        noMatchProbability: noMatch, confidence: confidence,
        actProbability: actProbability, inputTokenCount: 42)
}

@Test func acceptanceThresholdEqualityPassesAndPreservesRawSelection() throws {
    let selection = acceptanceSelection()
    let policy = try ToolDecisionAcceptancePolicy(minimumProbability: 0.75, minimumMargin: 0.625)
    let outcome = policy.evaluate(selection)
    #expect(outcome.status == .acceptedTool)
    #expect(outcome.reasons.isEmpty)
    #expect(outcome.selectedProbability == 0.75)
    #expect(outcome.probabilityMargin == 0.625)
    #expect(outcome.selection == selection)
    #expect(outcome.selection.selectedCandidate?.retrievalScore == -4)
    #expect(outcome.selection.confidence == 0.2)
    #expect(outcome.selection.actProbability == 0.1)
}

@Test func noMatchIsIncludedWhenComputingToolMargin() throws {
    let selection = acceptanceSelection(probabilityA: 0.5, probabilityB: 0.125, noMatch: 0.375)
    let outcome = try ToolDecisionAcceptancePolicy(minimumProbability: 0.5, minimumMargin: 0.25)
        .evaluate(selection)
    #expect(outcome.status == .abstained)
    #expect(outcome.reasons == [.belowMinimumMargin])
    #expect(outcome.selectedProbability == 0.5)
    #expect(outcome.probabilityMargin == 0.125)
    #expect(outcome.selection == selection)
    #expect(outcome.selection.selectedCandidateID == "a")
}

@Test func toolPolicyCanReportBothFailedGates() throws {
    let outcome = try ToolDecisionAcceptancePolicy(minimumProbability: 0.875, minimumMargin: 0.75)
        .evaluate(acceptanceSelection())
    #expect(outcome.status == .abstained)
    #expect(outcome.reasons == [.belowMinimumProbability, .belowMinimumMargin])
}

@Test func noMatchAnswerUsesTheSameAcceptanceGates() throws {
    let selection = acceptanceSelection(selectedID: nil, probabilityA: 0.125,
                                        probabilityB: 0.125, noMatch: 0.75)
    let accepted = try ToolDecisionAcceptancePolicy(minimumProbability: 0.75, minimumMargin: 0.625)
        .evaluate(selection)
    #expect(accepted.status == .acceptedNoMatch)
    #expect(accepted.selectedProbability == 0.75)
    #expect(accepted.probabilityMargin == 0.625)
    let uncertain = try ToolDecisionAcceptancePolicy(minimumProbability: 0.875, minimumMargin: 0)
        .evaluate(selection)
    #expect(uncertain.status == .abstained)
    #expect(uncertain.reasons == [.belowMinimumProbability])
    #expect(uncertain.selection == selection)
    #expect(uncertain.selection.selectedCandidateID == nil)
}

@Test func tiesAbstainOnlyWithAnEnabledMarginGate() throws {
    let selection = acceptanceSelection(probabilityA: 0.5, probabilityB: 0, noMatch: 0.5)
    let disabledMargin = try ToolDecisionAcceptancePolicy(minimumProbability: 0.5, minimumMargin: 0)
        .evaluate(selection)
    #expect(disabledMargin.status == .acceptedTool)
    #expect(disabledMargin.probabilityMargin == 0)
    let enabledMargin = try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: 0.01)
        .evaluate(selection)
    #expect(enabledMargin.status == .abstained)
    #expect(enabledMargin.reasons == [.belowMinimumMargin])
}

@Test func disabledGatesPreserveNonArgmaxAnswerAndNegativeMargin() throws {
    let selection = acceptanceSelection(selectedID: "b")
    let disabled = try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: 0)
        .evaluate(selection)
    #expect(disabled.status == .acceptedTool)
    #expect(disabled.selectedProbability == 0.125)
    #expect(disabled.probabilityMargin == -0.625)
    #expect(disabled.selection == selection)
    let gated = try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: 0.01)
        .evaluate(selection)
    #expect(gated.status == .abstained)
    #expect(gated.reasons == [.belowMinimumMargin])
    #expect(gated.probabilityMargin == -0.625)
}

@Test func zeroProbabilityGatePreservesZeroProbabilitySelectedAnswer() throws {
    let selection = acceptanceSelection(selectedID: "b", probabilityA: 0.75,
                                        probabilityB: 0, noMatch: 0.25)
    let outcome = try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: 0)
        .evaluate(selection)
    #expect(outcome.status == .acceptedTool)
    #expect(outcome.selectedProbability == 0)
}

@Test func emptyCandidatesRemainDeterministicNoMatchWithoutProbabilities() throws {
    let selection = ToolDecisionSelection(selectedCandidate: nil, candidateProbabilities: [],
        noMatchProbability: nil, confidence: nil, actProbability: nil, inputTokenCount: nil)
    let outcome = try ToolDecisionAcceptancePolicy(minimumProbability: 1, minimumMargin: 1)
        .evaluate(selection)
    #expect(outcome.status == .acceptedNoMatch)
    #expect(outcome.reasons.isEmpty)
    #expect(outcome.selectedProbability == nil)
    #expect(outcome.probabilityMargin == nil)
    #expect(outcome.selection == selection)
}

@Test(arguments: [Double.nan, Double.infinity, -Double.infinity, -0.1, 1.1])
func invalidToolAcceptanceThresholdsAreRejected(threshold: Double) {
    #expect(throws: DecisionError.self) {
        try ToolDecisionAcceptancePolicy(minimumProbability: threshold, minimumMargin: 0)
    }
    #expect(throws: DecisionError.self) {
        try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: threshold)
    }
}

@Test(arguments: [
    acceptanceSelection(selectedID: "unknown"),
    acceptanceSelection(probabilityA: 0.8),
    acceptanceSelection(probabilityA: .nan),
    acceptanceSelection(noMatch: .infinity),
    acceptanceSelection(actProbability: 1.1),
    ToolDecisionSelection(selectedCandidate: nil,
        candidateProbabilities: [.init(optionID: "a", probability: 0.5), .init(optionID: "a", probability: 0.25)],
        noMatchProbability: 0.25, confidence: nil, actProbability: nil, inputTokenCount: nil),
    ToolDecisionSelection(selectedCandidate: nil,
        candidateProbabilities: [.init(optionID: ToolDecisionSelector.noMatchID, probability: 0.75)],
        noMatchProbability: 0.25, confidence: nil, actProbability: nil, inputTokenCount: nil),
    ToolDecisionSelection(selectedCandidate: nil, candidateProbabilities: [],
        noMatchProbability: 1, confidence: nil, actProbability: nil, inputTokenCount: nil),
    ToolDecisionSelection(selectedCandidate: nil,
        candidateProbabilities: [.init(optionID: "a", probability: 1)],
        noMatchProbability: nil, confidence: nil, actProbability: nil, inputTokenCount: nil)
])
func malformedToolSelectionsAbstainWithoutInventingProbabilities(selection: ToolDecisionSelection) throws {
    let outcome = try ToolDecisionAcceptancePolicy(minimumProbability: 0, minimumMargin: 0)
        .evaluate(selection)
    #expect(outcome.status == .abstained)
    #expect(outcome.reasons == [.invalidSelection])
    #expect(outcome.selectedProbability == nil)
    #expect(outcome.probabilityMargin == nil)
    #expect(outcome.selection.selectedCandidateID == selection.selectedCandidateID)
}
