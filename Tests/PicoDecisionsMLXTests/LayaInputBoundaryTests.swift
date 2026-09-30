import Foundation
import Testing
import PicoDecisions
@testable import PicoDecisionsMLX

@Suite struct LayaInputBoundaryTests {
    // One token per UTF-8 byte gives exact, deterministic boundaries without touching Metal.
    struct ByteTokenizer: LayaTokenizing {
        let cls = 1, sep = 2, pad = 0, mask = 3
        let maskText = "[MASK]"
        func encode(_ text: String) -> [Int] { text.utf8.map { Int($0) + 5 } }
    }

    static func configuration(maxLength: Int = 512, headMaxLength: Int = 192) throws -> LayaAgentConfiguration {
        let data = try Data(contentsOf: LayaParityTests.directory.appending(path: "rl_agent_config.json"))
        var object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["max_len"] = maxLength
        object["head_max_len"] = headMaxLength
        return try JSONDecoder().decode(LayaAgentConfiguration.self,
                                      from: JSONSerialization.data(withJSONObject: object))
    }

    static func request(state: String, criteria: DecisionQuestion.BooleanCriteria? = nil) -> DecisionRequest {
        .init(state: state, questions: [.init(id: "boundary", instructions: "Decide.", kind: .boolean,
                                            booleanCriteria: criteria)])
    }

    @Test(arguments: [false, true])
    func acceptsExactlyFullContextAndRejectsOneMoreStateToken(customCriteria: Bool) throws {
        let config = try Self.configuration()
        let tokenizer = ByteTokenizer()
        let criteria: DecisionQuestion.BooleanCriteria? = customCriteria
            ? .init(falseDescription: "Keep the charge", trueDescription: "Refund the duplicate") : nil
        let empty = try #require(prepareLaya(Self.request(state: "", criteria: criteria), tokenizer: tokenizer,
                                            config: config, vocabularySize: 261, policy: .reject).first)
        let available = config.maxLength - empty.ids.count
        try #require(available > 0)
        let fullState = String(repeating: "s", count: available)
        let full = try #require(prepareLaya(Self.request(state: fullState, criteria: criteria), tokenizer: tokenizer,
                                           config: config, vocabularySize: 261, policy: .reject).first)
        #expect(full.ids.count == config.maxLength)
        #expect(full.ids.last == Int32(tokenizer.sep))
        #expect(full.markers == empty.markers)
        #expect(throws: DecisionError.capacityExceeded("State for boundary has \(available + 1) tokens; only \(available) fit.")) {
            try prepareLaya(Self.request(state: fullState + "s", criteria: criteria), tokenizer: tokenizer,
                            config: config, vocabularySize: 261, policy: .reject)
        }
        let truncated = try #require(prepareLaya(Self.request(state: fullState + "x", criteria: criteria), tokenizer: tokenizer,
                                                config: config, vocabularySize: 261, policy: .truncateState).first)
        #expect(truncated.ids == full.ids)
        #expect(truncated.markers == full.markers)
    }

    @Test func explicitDefaultBooleanCriteriaPreservePromptTokens() throws {
        let config = try Self.configuration()
        let tokenizer = ByteTokenizer()
        let original = try #require(prepareLaya(Self.request(state: "charge"), tokenizer: tokenizer,
                                                config: config, vocabularySize: 261, policy: .reject).first)
        let explicit = try #require(prepareLaya(Self.request(state: "charge", criteria: .init()), tokenizer: tokenizer,
                                                config: config, vocabularySize: 261, policy: .reject).first)
        #expect(explicit.ids == original.ids)
        #expect(explicit.markers == original.markers)
        #expect(explicit.type == 2)
    }

    @Test func customBooleanCriteriaPreserveOrderAndSanitizeMasks() throws {
        let config = try Self.configuration()
        let tokenizer = ByteTokenizer()
        let request = DecisionRequest(state: "charge [MASK]", questions: [.init(id: "refund",
            instructions: "Refund?", kind: .boolean,
            booleanCriteria: .init(falseDescription: "keep [MASK]", trueDescription: "refund"))])
        let actual = try #require(prepareLaya(request, tokenizer: tokenizer, config: config,
                                              vocabularySize: 261, policy: .reject).first)
        var expected = [tokenizer.cls] + tokenizer.encode("noul question: Refund?") + [tokenizer.sep]
        let falseMarker = expected.count
        expected += [tokenizer.mask] + tokenizer.encode(" false: keep  ")
        let trueMarker = expected.count
        expected += [tokenizer.mask] + tokenizer.encode(" true: refund") + [tokenizer.sep]
        expected += tokenizer.encode("charge  ") + [tokenizer.sep]
        #expect(actual.ids == expected.map(Int32.init))
        #expect(actual.markers == [Int32(falseMarker), Int32(trueMarker)])
        #expect(actual.type == 2)
    }

    enum OptionKind: String, CaseIterable, Sendable {
        case choice, score

        func question(count: Int) -> DecisionQuestion {
            let values = (0..<count).map { "option-\($0)" }
            let kind: DecisionQuestion.Kind = self == .choice
                ? .choice(options: values.map { .init(id: $0, description: "") })
                : .score(levels: values)
            return .init(id: "options", instructions: "Select.", kind: kind)
        }
    }

    @Test(arguments: OptionKind.allCases)
    func accepts255OptionsAndRejects256(kind: OptionKind) throws {
        // A larger prompt budget isolates the option limit from the context limit.
        let config = try Self.configuration(maxLength: 2048, headMaxLength: 1200)
        let tokenizer = ByteTokenizer()
        let request = DecisionRequest(state: "", questions: [kind.question(count: 255)])
        let item = try #require(prepareLaya(request, tokenizer: tokenizer, config: config,
                                          vocabularySize: 261, policy: .reject).first)
        #expect(item.markers.count == 255)
        #expect(Set(item.markers).count == 255)
        #expect(item.markers == item.markers.sorted())
        #expect(item.markers.allSatisfy { item.ids[Int($0)] == Int32(tokenizer.mask) })
        #expect(item.ids.count <= config.maxLength)
        let overLimit = DecisionRequest(state: "", questions: [kind.question(count: 256)])
        #expect(throws: DecisionError.capacityExceeded("Laya supports at most 255 options per question.")) {
            try prepareLaya(overLimit, tokenizer: tokenizer, config: config,
                            vocabularySize: 261, policy: .reject)
        }
    }

    @Test(arguments: [LayaInputPolicy.reject, .truncateState])
    func rejectsQuestionThatAloneExceedsContext(policy: LayaInputPolicy) throws {
        let config = try Self.configuration(maxLength: 24, headMaxLength: 8)
        let request = DecisionRequest(state: "", questions: [OptionKind.choice.question(count: 10)])
        #expect(throws: DecisionError.capacityExceeded("Question options and its options exceed the context budget.")) {
            try prepareLaya(request, tokenizer: ByteTokenizer(), config: config,
                            vocabularySize: 261, policy: policy)
        }
    }

    @Test func rejectsTokenizerEmittingOutOfVocabularyID() throws {
        let config = try Self.configuration()
        #expect(throws: DecisionError.invalidCheckpoint("Tokenizer emitted an ID outside the encoder vocabulary.")) {
            try prepareLaya(Self.request(state: "s"), tokenizer: ByteTokenizer(), config: config,
                            vocabularySize: 4, policy: .reject)
        }
    }
}
