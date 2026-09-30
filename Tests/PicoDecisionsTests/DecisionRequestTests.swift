import Testing
@testable import PicoDecisions

@Test func rejectsAmbiguousIdentifiers() {
    let q = DecisionQuestion(id: "route", instructions: "Pick a route", kind: .boolean)
    #expect(throws: DecisionError.self) {
        try DecisionRequest(state: "", questions: [q, q]).validate()
    }
    let repeated = DecisionQuestion(id: "route", instructions: "Pick a route", kind: .choice(options: [
        .init(id: "same", description: "A"), .init(id: "same", description: "B")
    ]))
    #expect(throws: DecisionError.self) {
        try DecisionRequest(state: "", questions: [repeated]).validate()
    }
}

@Test func rejectsEmptyChoicesAndRubrics() {
    for kind in [DecisionQuestion.Kind.choice(options: []), .score(levels: [])] {
        #expect(throws: DecisionError.self) {
            try DecisionRequest(state: "", questions: [.init(id: "q", instructions: "Choose", kind: kind)]).validate()
        }
    }
}

@Test func acceptsEmptyStateAndNoQuestions() throws {
    try DecisionRequest(state: "", questions: []).validate()
    try DecisionRequest(state: "", questions: [.init(id: "q", instructions: "Is it empty?", kind: .boolean)]).validate()
}
