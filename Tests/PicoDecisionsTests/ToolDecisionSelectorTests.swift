import Testing
@testable import PicoDecisions

private actor RecordingToolDecisionModel: DecisionModel {
    let results: [DecisionResult]
    let cancelDuringPrediction: Bool
    private(set) var requests: [DecisionRequest] = []

    init(results: [DecisionResult], cancelDuringPrediction: Bool = false) {
        self.results = results
        self.cancelDuringPrediction = cancelDuringPrediction
    }

    func predict(_ request: DecisionRequest) async throws -> [DecisionResult] {
        requests.append(request)
        if cancelDuringPrediction { withUnsafeCurrentTask { $0?.cancel() } }
        return results
    }
}

private let toolCandidates = [
    ToolDecisionCandidate(id: "mcp.calendar/list_events", description: "List calendar events", retrievalScore: 12.5),
    ToolDecisionCandidate(id: "weather", description: "Fetch weather forecasts", retrievalScore: -0.2)
]

private func toolChoice(selectedID: String = "weather", probabilities: [OptionProbability]? = nil,
                        confidence: Double? = 0.42, actProbability: Double? = 0.8,
                        inputTokenCount: Int? = 57) -> DecisionResult {
    DecisionResult(id: ToolDecisionSelector.questionID,
                   answer: .choice(selectedID: selectedID, probabilities: probabilities ?? [
                    .init(optionID: "weather", probability: 0.6),
                    .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.1),
                    .init(optionID: "mcp.calendar/list_events", probability: 0.3)
                   ]), confidence: confidence, actProbability: actProbability,
                   inputTokenCount: inputTokenCount)
}

@Test func toolDecisionPreservesIdentityAndKeepsRetrievalSeparate() async throws {
    let model = RecordingToolDecisionModel(results: [toolChoice()])
    let selector = try ToolDecisionSelector(model: model)
    let result = try await selector.select(query: "Will it rain tomorrow?", candidates: toolCandidates)
    #expect(result.selectedCandidate == toolCandidates[1])
    #expect(result.selectedCandidateID == "weather")
    #expect(result.candidateProbabilities == [
        .init(optionID: "mcp.calendar/list_events", probability: 0.3),
        .init(optionID: "weather", probability: 0.6)
    ])
    #expect(result.noMatchProbability == 0.1)
    #expect(result.confidence == 0.42)
    #expect(result.actProbability == 0.8)
    #expect(result.inputTokenCount == 57)
    let request = try #require(await model.requests.first)
    #expect(request.state == "Will it rain tomorrow?")
    let question = try #require(request.questions.first)
    guard case .choice(let options) = question.kind else {
        Issue.record("Expected a tool choice question")
        return
    }
    #expect(options == toolCandidates.map { DecisionOption(id: $0.id, description: $0.description) } + [
        .init(id: ToolDecisionSelector.noMatchID,
              description: "No matching tool: none of the candidate tools is relevant to the request.")
    ])
}

@Test func toolDecisionCanSelectNoMatch() async throws {
    let model = RecordingToolDecisionModel(results: [toolChoice(selectedID: ToolDecisionSelector.noMatchID,
        probabilities: [.init(optionID: "mcp.calendar/list_events", probability: 0.1),
                        .init(optionID: "weather", probability: 0.2),
                        .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.7)])])
    let result = try await ToolDecisionSelector(model: model).select(query: "Play some music", candidates: toolCandidates)
    #expect(result.selectedCandidate == nil)
    #expect(result.noMatchProbability == 0.7)
}

@Test func toolDecisionLeavesConfidenceAndActionPolicyToCaller() async throws {
    let model = RecordingToolDecisionModel(results: [toolChoice(confidence: 0, actProbability: 0)])
    let result = try await ToolDecisionSelector(model: model).select(query: "Weather?", candidates: toolCandidates)
    #expect(result.selectedCandidateID == "weather")
    #expect(result.confidence == 0)
    #expect(result.actProbability == 0)
}

@Test func emptyToolCandidatesSkipInference() async throws {
    let model = RecordingToolDecisionModel(results: [])
    let result = try await ToolDecisionSelector(model: model).select(query: "", candidates: [])
    #expect(result.selectedCandidate == nil)
    #expect(result.candidateProbabilities.isEmpty)
    #expect(result.noMatchProbability == nil)
    #expect(result.confidence == nil)
    #expect(result.actProbability == nil)
    #expect(result.inputTokenCount == nil)
    #expect(await model.requests.isEmpty)
}

@Test(arguments: [0, -1, 255, Int.max])
func invalidToolDecisionCapacityIsRejected(capacity: Int) {
    #expect(throws: DecisionError.self) {
        try ToolDecisionSelector(model: RecordingToolDecisionModel(results: []), maximumCandidates: capacity)
    }
}

@Test func maximumToolCandidateCountLeavesRoomForNoMatch() async throws {
    let candidates = (0..<254).map { ToolDecisionCandidate(id: "tool_\($0)", description: "Tool \($0)") }
    let probabilities = candidates.map { OptionProbability(optionID: $0.id, probability: 0) }
        + [OptionProbability(optionID: ToolDecisionSelector.noMatchID, probability: 1)]
    let model = RecordingToolDecisionModel(results: [toolChoice(selectedID: ToolDecisionSelector.noMatchID,
        probabilities: probabilities, confidence: nil, actProbability: nil, inputTokenCount: nil)])
    let result = try await ToolDecisionSelector(model: model, maximumCandidates: 254)
        .select(query: "query", candidates: candidates)
    #expect(result.selectedCandidate == nil)
    #expect(result.candidateProbabilities.count == 254)
    #expect(result.noMatchProbability == 1)
    let request = try #require(await model.requests.first)
    let question = try #require(request.questions.first)
    guard case .choice(let options) = question.kind else {
        Issue.record("Expected a tool choice question")
        return
    }
    #expect(options.count == 255)
}

@Test func excessToolCandidatesFailBeforeModelCall() async throws {
    let model = RecordingToolDecisionModel(results: [])
    let selector = try ToolDecisionSelector(model: model, maximumCandidates: 1)
    await #expect(throws: DecisionError.self) {
        try await selector.select(query: "query", candidates: toolCandidates)
    }
    #expect(await model.requests.isEmpty)
}

@Test(arguments: [
    [ToolDecisionCandidate(id: "same", description: "A"), .init(id: "same", description: "B")],
    [ToolDecisionCandidate(id: ToolDecisionSelector.noMatchID, description: "Reserved")],
    [ToolDecisionCandidate(id: " \n", description: "Blank identifier")],
    [ToolDecisionCandidate(id: "empty", description: " \n")],
    [ToolDecisionCandidate(id: "score", description: "Invalid retrieval score", retrievalScore: .nan)]
])
func invalidToolCandidatesFailBeforeModelCall(candidates: [ToolDecisionCandidate]) async throws {
    let model = RecordingToolDecisionModel(results: [])
    let selector = try ToolDecisionSelector(model: model)
    await #expect(throws: DecisionError.self) {
        try await selector.select(query: "query", candidates: candidates)
    }
    #expect(await model.requests.isEmpty)
}

@Test func blankToolQueryFailsBeforeModelCall() async throws {
    let model = RecordingToolDecisionModel(results: [])
    let selector = try ToolDecisionSelector(model: model)
    await #expect(throws: DecisionError.self) {
        try await selector.select(query: " \n", candidates: toolCandidates)
    }
    #expect(await model.requests.isEmpty)
}

@Test(arguments: [
    [],
    [toolChoice(), toolChoice()],
    [DecisionResult(id: "other", answer: .boolean(probabilityTrue: 0.5))],
    [DecisionResult(id: ToolDecisionSelector.questionID, answer: .boolean(probabilityTrue: 0.5))],
    [toolChoice(selectedID: "unknown")],
    [toolChoice(probabilities: [.init(optionID: "weather", probability: 1)])],
    [toolChoice(probabilities: [.init(optionID: "weather", probability: 0.3),
                                .init(optionID: "weather", probability: 0.3),
                                .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.4)])],
    [toolChoice(probabilities: [.init(optionID: "unknown", probability: 0.3),
                                .init(optionID: "weather", probability: 0.3),
                                .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.4)])],
    [toolChoice(probabilities: [.init(optionID: "mcp.calendar/list_events", probability: -0.1),
                                .init(optionID: "weather", probability: 0.7),
                                .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.4)])],
    [toolChoice(probabilities: [.init(optionID: "mcp.calendar/list_events", probability: 0.1),
                                .init(optionID: "weather", probability: 1.1),
                                .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.1)])],
    [toolChoice(probabilities: [.init(optionID: "mcp.calendar/list_events", probability: 0.1),
                                .init(optionID: "weather", probability: 0.1),
                                .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.1)])],
    [toolChoice(actProbability: 1.1)],
    [toolChoice(inputTokenCount: -1)]
])
func malformedToolDecisionOutputIsRejected(results: [DecisionResult]) async throws {
    let selector = try ToolDecisionSelector(model: RecordingToolDecisionModel(results: results))
    await #expect(throws: ToolDecisionError.self) {
        try await selector.select(query: "query", candidates: toolCandidates)
    }
}

@Test(arguments: [
    toolChoice(probabilities: [.init(optionID: "mcp.calendar/list_events", probability: 0.3),
                              .init(optionID: "weather", probability: .nan),
                              .init(optionID: ToolDecisionSelector.noMatchID, probability: 0.7)]),
    toolChoice(confidence: .infinity),
    toolChoice(actProbability: .nan)
])
func nonFiniteToolDecisionOutputIsRejected(result: DecisionResult) async throws {
    let selector = try ToolDecisionSelector(model: RecordingToolDecisionModel(results: [result]))
    await #expect(throws: DecisionError.nonFiniteOutput) {
        try await selector.select(query: "query", candidates: toolCandidates)
    }
}

@Test func toolDecisionHonorsCancellationBeforeInference() async throws {
    let model = RecordingToolDecisionModel(results: [toolChoice()])
    let selector = try ToolDecisionSelector(model: model)
    let task = Task {
        withUnsafeCurrentTask { $0?.cancel() }
        return try await selector.select(query: "query", candidates: toolCandidates)
    }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await model.requests.isEmpty)
}

@Test func toolDecisionHonorsCancellationAfterInference() async throws {
    let model = RecordingToolDecisionModel(results: [toolChoice()], cancelDuringPrediction: true)
    let selector = try ToolDecisionSelector(model: model)
    let task = Task { try await selector.select(query: "query", candidates: toolCandidates) }
    await #expect(throws: CancellationError.self) { try await task.value }
    #expect(await model.requests.count == 1)
}
