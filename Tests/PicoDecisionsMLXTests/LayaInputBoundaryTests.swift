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
        #expect(truncated.inputDiagnostics?.state == .init(
            originalTokenCount: available + 1, retainedTokenCount: available))
        #expect(truncated.inputDiagnostics?.state.wasTruncated == true)
        #expect(full.inputDiagnostics?.state.wasTruncated == false)
    }

    @Test func reportsUntruncatedNormalizedComponentsAndPropagatesToResult() throws {
        let config = try Self.configuration()
        let tokenizer = ByteTokenizer()
        let request = Self.request(state: "hello[MASK]world", criteria: .init(
            falseDescription: "keep", trueDescription: "act"))
        let item = try #require(prepareLaya(request, tokenizer: tokenizer, config: config,
                                          vocabularySize: 261, policy: .reject).first)
        let diagnostics = try #require(item.inputDiagnostics)
        #expect(!diagnostics.wasTruncated)
        #expect(diagnostics.state == .init(originalTokenCount: 11, retainedTokenCount: 11))
        #expect(diagnostics.instructions == .init(originalTokenCount: 22, retainedTokenCount: 22))
        #expect(diagnostics.options == [
            .init(optionID: "false", tokens: .init(originalTokenCount: 12, retainedTokenCount: 12)),
            .init(optionID: "true", tokens: .init(originalTokenCount: 10, retainedTokenCount: 10))
        ])
        // Four CLS/SEP tokens and one MASK per option are structural.
        #expect(item.ids.count == 11 + 22 + 12 + 10 + 4 + 2)
        let result = try layaResult(item, logits: [0, 1], action: [1, 0], config: config)
        #expect(result.inputDiagnostics == diagnostics)
        #expect(result.inputTokenCount == item.ids.count)
    }

    @Test func reportsOptionPrefixLimitWithStableChoiceIDsUnderRejectPolicy() throws {
        let config = try Self.configuration()
        let tokenizer = ByteTokenizer()
        let request = DecisionRequest(state: "state", questions: [.init(id: "choice", instructions: "Pick.",
            kind: .choice(options: [.init(id: "first", description: String(repeating: "x", count: 70)),
                                     .init(id: "second", description: "short")]))])
        let item = try #require(prepareLaya(request, tokenizer: tokenizer, config: config,
                                          vocabularySize: 261, policy: .reject).first)
        let diagnostics = try #require(item.inputDiagnostics)
        #expect(diagnostics.wasTruncated)
        #expect(!diagnostics.state.wasTruncated)
        #expect(!diagnostics.instructions.wasTruncated)
        #expect(diagnostics.options == [
            .init(optionID: "first", tokens: .init(originalTokenCount: 78, retainedTokenCount: 48)),
            .init(optionID: "second", tokens: .init(originalTokenCount: 14, retainedTokenCount: 14))
        ])
        #expect(Int(item.markers[1] - item.markers[0]) == 49)
    }

    @Test func reportsFinalHeadRebudgetingAndInstructionTruncation() throws {
        let config = try Self.configuration(headMaxLength: 64)
        let tokenizer = ByteTokenizer()
        let request = DecisionRequest(state: "state", questions: [.init(id: "score",
            instructions: String(repeating: "i", count: 100),
            kind: .score(levels: (0..<3).map { _ in String(repeating: "x", count: 70) }))])
        let item = try #require(prepareLaya(request, tokenizer: tokenizer, config: config,
                                          vocabularySize: 261, policy: .reject).first)
        let diagnostics = try #require(item.inputDiagnostics)
        #expect(diagnostics.instructions == .init(originalTokenCount: 116, retainedTokenCount: 16))
        #expect(diagnostics.options == (0..<3).map {
            .init(optionID: String($0), tokens: .init(originalTokenCount: 80, retainedTokenCount: 15))
        })
        #expect(diagnostics.wasTruncated)
        #expect(!diagnostics.state.wasTruncated)
        #expect(item.markers == [18, 34, 50])
        var expected = [tokenizer.cls] + tokenizer.encode("score question: " + String(repeating: "i", count: 100)).prefix(16)
            + [tokenizer.sep]
        for index in 0..<3 {
            expected += [tokenizer.mask] + tokenizer.encode(" level \(index): " + String(repeating: "x", count: 70)).prefix(15)
        }
        expected += [tokenizer.sep] + tokenizer.encode("state") + [tokenizer.sep]
        #expect(item.ids == expected.map(Int32.init))
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
