import Foundation
import MLX
import MLXNN
import Testing
import Tokenizers
import PicoDecisions
@testable import PicoDecisionsMLX

// These tests share the Metal runtime. Serialize model construction, inference, and teardown.
@Suite(.serialized)
struct LayaParityTests {
    static let directory = Bundle.module.resourceURL!.appending(path: "Fixtures/tiny")

    struct Item: Decodable {
        let ids: [Int32]
        let markers: [Int32]
        let qtype: Int32
    }

    struct Reference: Decodable {
        let state: String
        let items: [Item]
        let logits: [[Float]]
        let action_logits: [[Float]]
        let result: Result
        struct Result: Decodable {
            let answers: [String: Answer]
        }
        struct Answer: Decodable {
            let confidence: Double
            let choice: String?
            let score: Double?
            let noul: Double?
            let probabilities: [String: Double]?
            let action: Action
            struct Action: Decodable { let act_probability: Double }
        }
    }

    static func reference() throws -> Reference {
        try JSONDecoder().decode(Reference.self, from: Data(contentsOf: directory.appending(path: "reference.json")))
    }

    static func request(state: String) -> DecisionRequest {
        .init(state: state, questions: [
            .init(id: "route", instructions: "Choose the requested action.", kind: .choice(options: [
                .init(id: "lookup", description: "Find an order"),
                .init(id: "cancel", description: "Cancel an order"),
                .init(id: "none", description: "Neither action")
            ])),
            .init(id: "truth", instructions: "Does the user request cancellation?", kind: .boolean),
            .init(id: "priority", instructions: "Rate urgency.", kind: .score(levels: ["low", "medium", "high"])),
            .init(id: "single", instructions: "Select the option.", kind: .choice(options: [
                .init(id: "only", description: "The only option")
            ]))
        ])
    }

    @Test func exactTokenizationAndPromptParity() async throws {
        let r = try Self.reference()
        let directory = Self.directory
        let backend = try await AutoTokenizer.from(modelFolder: directory.appending(path: "tokenizer"))
        let tokenizer = try LayaTokenizer(tokenizer: backend,
            configuration: Data(contentsOf: directory.appending(path: "tokenizer/tokenizer_config.json")))
        let config = try JSONDecoder().decode(LayaAgentConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        let items = try prepareLaya(Self.request(state: r.state), tokenizer: tokenizer, config: config,
                                    vocabularySize: 261, policy: .reject)
        #expect(items.count == r.items.count)
        for (actual, expected) in zip(items, r.items) {
            #expect(actual.ids == expected.ids)
            #expect(actual.markers == expected.markers)
            #expect(actual.type == expected.qtype)
        }
        let long = Self.request(state: String(repeating: "long ", count: 1000))
        #expect(throws: DecisionError.self) {
            try prepareLaya(long, tokenizer: tokenizer, config: config, vocabularySize: 261, policy: .reject)
        }
        let truncated = try prepareLaya(long, tokenizer: tokenizer, config: config,
                                        vocabularySize: 261, policy: .truncateState)
        #expect(truncated.allSatisfy { $0.ids.count == config.maxLength && $0.ids.last == Int32(tokenizer.sep) })
    }

    @Test func rawNetworkMatchesPython() throws {
        try Stream.withNewDefaultStream {
            let r = try Self.reference()
            let decoder = JSONDecoder()
            let encoder = try decoder.decode(LayaEncoderConfiguration.self,
                from: Data(contentsOf: Self.directory.appending(path: "encoder/config.json")))
            let agent = try decoder.decode(LayaAgentConfiguration.self,
                from: Data(contentsOf: Self.directory.appending(path: "rl_agent_config.json")))
            let model = LayaNetwork(encoder: encoder, agent: agent)
            let weights = try loadArrays(url: Self.directory.appending(path: "model.safetensors"))
            try model.update(parameters: .unflattened(weights), verify: .all)
            model.train(false)
            let questions = Self.request(state: r.state).questions
            let items = zip(questions, r.items).map {
                LayaPreparedQuestion(question: $0.0, ids: $0.1.ids, markers: $0.1.markers, type: $0.1.qtype)
            }
            let batch = LayaBatch(items, pad: 0)
            let (logits, action) = model(ids: batch.ids, mask: batch.mask, positions: batch.positions,
                                         markerMask: batch.markerMask, types: batch.types)
            let pairs = [(logits.asArray(Float.self), r.logits.flatMap { $0 }),
                         (action.asArray(Float.self), r.action_logits.flatMap { $0 })]
            for (actual, expected) in pairs {
                #expect(actual.count == expected.count)
                let drift = zip(actual, expected).map { abs($0 - $1) }.max() ?? .infinity
                #expect(drift < 3e-5, "Maximum raw-logit difference: \(drift)")
            }
        }
    }

    @Test(arguments: [LayaPrecision.float32, .float16])
    func publicPredictionsMatchPython(precision: LayaPrecision) async throws {
        let r = try Self.reference()
        let model = try await LayaModel.load(from: Self.directory, precision: precision, batchSize: 2)
        let request = Self.request(state: r.state)
        let results = try await model.predict(request)
        let repeated = try await model.predict(request)
        #expect(results == repeated)
        #expect(results.map(\.id) == request.questions.map(\.id))
        for result in results {
            let expected = try #require(r.result.answers[result.id])
            #expect(abs(try #require(result.confidence) - expected.confidence) < 0.002)
            #expect(abs(try #require(result.actProbability) - expected.action.act_probability) < 0.002)
            switch result.answer {
            case .choice(let selected, let p):
                if precision == .float32 {
                    #expect(selected == expected.choice)
                } else {
                    // Random weights produce near ties (<1e-5 in logits); FP16 may reorder them.
                    let expectedP = try #require(expected.probabilities)
                    #expect(abs((expectedP[selected] ?? -.infinity) - (expectedP.values.max() ?? .infinity)) < 0.002)
                }
                for probability in p {
                    let expectedP = try #require(expected.probabilities?[probability.optionID])
                    #expect(abs(probability.probability - expectedP) < 0.002)
                }
                #expect(abs(p.reduce(0) { $0 + $1.probability } - 1) < 1e-10)
            case .score(let score, _): #expect(abs(score - (expected.score ?? .infinity)) < 0.002)
            case .boolean(let p): #expect(abs(p - (expected.noul ?? .infinity)) < 0.002)
            }
        }
        let empty = try await model.predict(.init(state: "", questions: []))
        #expect(empty.isEmpty)
    }

    @Test func inclusiveLocalWindowAndPaddedRows() {
        Stream.withNewDefaultStream {
            let mask = MLXArray(Array(repeating: Int32(1), count: 140)).reshaped(1, 140)
            let local = layaAttentionMasks(mask, window: 128).local.asArray(Bool.self)
            #expect(local[70 * 140 + 6])
            #expect(local[70 * 140 + 134])
            #expect(!local[70 * 140 + 5])
            #expect(!local[70 * 140 + 135])
            let padded = MLXArray([Int32](repeating: 1, count: 13) + [Int32](repeating: 0, count: 132)).reshaped(1, 145)
            let masks = layaAttentionMasks(padded, window: 128)
            let values = masks.local.asArray(Bool.self)
            #expect(values[144 * 145])
            #expect(!values[144 * 145 + 144])
        }
    }

    @Test func rejectsWeightNameCollisions() throws {
        try Stream.withNewDefaultStream {
            let value = MLXArray([Float(1)])
            #expect(throws: DecisionError.self) {
                try LayaModel.sanitize(["scorer.0.weight": value, "scorer.layers.0.weight": value])
            }
            let model = LayaNetwork(
                encoder: try JSONDecoder().decode(LayaEncoderConfiguration.self,
                    from: Data(contentsOf: Self.directory.appending(path: "encoder/config.json"))),
                agent: try JSONDecoder().decode(LayaAgentConfiguration.self,
                    from: Data(contentsOf: Self.directory.appending(path: "rl_agent_config.json"))))
            #expect(throws: (any Error).self) { try model.update(parameters: .unflattened([:]), verify: .all) }
        }
    }
}
