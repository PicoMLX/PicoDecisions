import Foundation
import MLX
import MLXNN
import Testing
import Tokenizers
import PicoDecisions
@testable import PicoDecisionsMLX

/// Opt-in tests: supply a local, unmodified multilingual checkpoint via the environment.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["PICODECISIONS_LAYA_MODEL"] != nil))
struct FullCheckpointTests {
    struct Fixture: Decodable { let cases: [Case] }
    struct Case: Decodable {
        let name: String
        let state: String
        let questions: [Question]
        let items: [LayaParityTests.Item]
        let logits: [[Float]]
        let action_logits: [[Float]]
        let result: LayaParityTests.Reference.Result
        var request: DecisionRequest { .init(state: state, questions: questions.map(\.question)) }
    }
    struct Question: Decodable {
        let id: String
        let instructions: String
        let type: String
        let options: [Option]?
        let levels: [String]?
        struct Option: Decodable { let id: String; let description: String }
        var question: DecisionQuestion {
            let kind: DecisionQuestion.Kind
            switch type {
            case "choice": kind = .choice(options: options!.map { .init(id: $0.id, description: $0.description) })
            case "score": kind = .score(levels: levels!)
            default: kind = .boolean
            }
            return .init(id: id, instructions: instructions, kind: kind)
        }
    }
    static func fixture() throws -> Fixture {
        try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
            Bundle.module.resourceURL!.appending(path: "Fixtures/multilingual-reference.json")))
    }
    static var directory: URL {
        URL(filePath: ProcessInfo.processInfo.environment["PICODECISIONS_LAYA_MODEL"]!)
    }

    @Test func multilingualTokensAndFP32Logits() async throws {
        let directory = Self.directory
        let decoder = JSONDecoder()
        let encoder = try decoder.decode(LayaEncoderConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "encoder/config.json")))
        let agent = try decoder.decode(LayaAgentConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        let backend = try await AutoTokenizer.from(modelFolder: directory.appending(path: "tokenizer"))
        let tokenizer = try LayaTokenizer(tokenizer: backend,
            configuration: Data(contentsOf: directory.appending(path: "tokenizer/tokenizer_config.json")))
        let cases = try Self.fixture().cases
        try Stream.withNewDefaultStream {
            let model = LayaNetwork(encoder: encoder, agent: agent)
            let weights = try LayaModel.sanitize(loadArrays(url: directory.appending(path: "model.safetensors")))
            try model.update(parameters: .unflattened(weights.mapValues { $0.asType(.float32) }), verify: .all)
            model.train(false)
            var maxLogit: Float = 0, maxAction: Float = 0
            for c in cases {
                let items = try prepareLaya(c.request, tokenizer: tokenizer, config: agent,
                    vocabularySize: encoder.vocabSize, policy: .truncateState)
                for (actual, expected) in zip(items, c.items) {
                    #expect(actual.ids == expected.ids, "Token IDs: \(c.name)")
                    #expect(actual.markers == expected.markers, "Markers: \(c.name)")
                    #expect(actual.type == expected.qtype)
                }
                let batch = LayaBatch(items, pad: Int32(tokenizer.pad))
                let (logits, action) = model(ids: batch.ids, mask: batch.mask,
                    positions: batch.positions, markerMask: batch.markerMask, types: batch.types)
                let actual = logits.asArray(Float.self), actions = action.asArray(Float.self)
                let expected = c.logits.flatMap { $0 }, expectedActions = c.action_logits.flatMap { $0 }
                #expect(actual.count == expected.count)
                #expect(actions.count == expectedActions.count)
                let logitDrift = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
                let actionDrift = zip(actions, expectedActions).map { abs($0 - $1) }.max() ?? .infinity
                maxLogit = max(maxLogit, logitDrift); maxAction = max(maxAction, actionDrift)
                #expect(logitDrift < 0.001, "\(c.name) raw-logit drift: \(logitDrift)")
                #expect(actionDrift < 0.01, "\(c.name) action-logit drift: \(actionDrift)")
            }
            print("Multilingual FP32 max logit drift: \(maxLogit); action drift: \(maxAction)")
        }
    }

    @Test(arguments: [LayaPrecision.float32, .float16])
    func multilingualPublicResults(precision: LayaPrecision) async throws {
        let model = try await LayaModel.load(from: Self.directory, precision: precision,
            batchSize: 8, inputPolicy: .truncateState)
        let tolerance = precision == .float32 ? 0.0002 : 0.005
        var maxDifference: Double = 0
        for c in try Self.fixture().cases {
            let results = try await model.predict(c.request)
            #expect(results.map(\.id) == c.questions.map(\.id))
            for (result, item) in zip(results, c.items) {
                let expected = try #require(c.result.answers[result.id])
                #expect(result.inputTokenCount == item.ids.count)
                var differences = [abs(try #require(result.confidence) - expected.confidence),
                    abs(try #require(result.actProbability) - expected.action.act_probability)]
                switch result.answer {
                case .choice(let selected, let probabilities):
                    #expect(selected == expected.choice, "\(c.name)")
                    for p in probabilities {
                        differences.append(abs(p.probability - (expected.probabilities?[p.optionID] ?? .infinity)))
                    }
                case .score(let value, let probabilities):
                    differences.append(abs(value - (expected.score ?? .infinity)))
                    for (i, p) in probabilities.enumerated() {
                        differences.append(abs(p - (expected.probabilities?[String(i)] ?? .infinity)))
                    }
                case .boolean(let value): differences.append(abs(value - (expected.noul ?? .infinity)))
                }
                let difference = differences.max() ?? .infinity
                maxDifference = max(maxDifference, difference)
                #expect(difference < tolerance, "\(c.name)/\(result.id) \(precision): \(difference)")
            }
        }
        print("Multilingual \(precision) max output difference: \(maxDifference)")
    }
}
