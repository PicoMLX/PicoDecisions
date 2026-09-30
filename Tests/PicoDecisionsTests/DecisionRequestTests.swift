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

@Test func acceptsCustomAndPartialBooleanCriteria() throws {
    let custom = DecisionQuestion.BooleanCriteria(falseDescription: "Keep the charge",
                                                 trueDescription: "Refund the duplicate charge")
    let trueOnly = DecisionQuestion.BooleanCriteria(trueDescription: "Refund the charge")
    let falseOnly = DecisionQuestion.BooleanCriteria(falseDescription: "Keep the charge")
    #expect(trueOnly.falseDescription == "no, the statement does not hold")
    #expect(falseOnly.trueDescription == "yes, the statement holds")
    for criteria in [custom, trueOnly, falseOnly, .init()] {
        try DecisionRequest(state: "", questions: [.init(id: "refund", instructions: "Refund?",
            kind: .boolean, booleanCriteria: criteria)]).validate()
    }
}

@Test func rejectsBlankBooleanCriteria() {
    for criteria in [DecisionQuestion.BooleanCriteria(falseDescription: " \n"),
                     .init(trueDescription: "\t")] {
        #expect(throws: DecisionError.invalidRequest("Boolean criteria need nonblank false and true descriptions.")) {
            try DecisionRequest(state: "", questions: [.init(id: "refund", instructions: "Refund?",
                kind: .boolean, booleanCriteria: criteria)]).validate()
        }
    }
}

@Test func rejectsBooleanCriteriaOnOtherQuestionKinds() {
    for kind in [DecisionQuestion.Kind.choice(options: [.init(id: "a", description: "A")]),
                 .score(levels: ["low", "high"])] {
        #expect(throws: DecisionError.invalidRequest("Boolean criteria require a boolean question.")) {
            try DecisionRequest(state: "", questions: [.init(id: "q", instructions: "Choose.",
                kind: kind, booleanCriteria: .init(trueDescription: "A custom true answer"))]).validate()
        }
    }
}
