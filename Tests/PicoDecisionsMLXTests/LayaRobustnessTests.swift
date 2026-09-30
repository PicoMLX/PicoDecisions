import Foundation
import Testing
import PicoDecisions
@testable import PicoDecisionsMLX

// Keep model loading and inference in the existing serialized Metal test suite.
extension LayaParityTests {
    @Test func cancelledLoadStopsBeforeReadingCheckpoint() async throws {
        let checkpoint = TemporaryLayaCheckpoint.nonexistentDirectory()
        let load = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try await LayaModel.load(from: checkpoint)
        }
        await #expect(throws: CancellationError.self) { try await load.value }
    }

    @Test func cancelledPredictionLeavesModelUsable() async throws {
        let model = try await LayaModel.load(from: Self.directory, precision: .float32, batchSize: 1)
        let request = Self.request(state: "Please cancel my order.")
        let expected = try await model.predict(request)
        let prediction = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.predict(request)
        }
        await #expect(throws: CancellationError.self) { try await prediction.value }
        #expect(try await model.predict(request) == expected)
    }

    @Test func concurrentCallersPreserveResultsAndRequestOrder() async throws {
        let model = try await LayaModel.load(from: Self.directory, precision: .float32, batchSize: 1)
        let requests = [
            "Find order 123.", "Cancel my order, please.", "Hello.", "Please help me with an urgent order."
        ].enumerated().map { index, state in
            let questions = Self.request(state: state).questions.map {
                DecisionQuestion(id: "caller-\(index)-\($0.id)", instructions: $0.instructions, kind: $0.kind)
            }
            return DecisionRequest(state: state, questions: questions)
        }
        var expected: [[DecisionResult]] = []
        for request in requests { expected.append(try await model.predict(request)) }

        let concurrent = try await withThrowingTaskGroup(of: (Int, [DecisionResult]).self) { group in
            for (index, request) in requests.enumerated() {
                group.addTask { (index, try await model.predict(request)) }
            }
            var results: [Int: [DecisionResult]] = [:]
            for try await (index, result) in group { results[index] = result }
            return results
        }
        #expect(concurrent.count == requests.count)
        for index in requests.indices {
            let results = try #require(concurrent[index])
            #expect(results == expected[index])
            #expect(results.map(\.id) == requests[index].questions.map(\.id))
        }
    }

    @Test(arguments: [0, 65])
    func rejectsInvalidBatchSizeBeforeReadingCheckpoint(batchSize: Int) async {
        await #expect(throws: DecisionError.invalidConfiguration("Batch size must be between 1 and 64.")) {
            try await LayaModel.load(from: TemporaryLayaCheckpoint.nonexistentDirectory(), batchSize: batchSize)
        }
    }

    @Test func rejectsRemoteCheckpointURL() async throws {
        let directory = try #require(URL(string: "https://example.invalid/checkpoint"))
        await #expect(throws: DecisionError.invalidCheckpoint("Supply a local checkpoint directory.")) {
            try await LayaModel.load(from: directory)
        }
    }

    @Test func rejectsMalformedEncoderJSON() async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try Data("not JSON".utf8).write(to: checkpoint.directory.appending(path: "encoder/config.json"))
        await #expect(throws: DecodingError.self) { try await LayaModel.load(from: checkpoint.directory) }
    }

    @Test func rejectsAgentContextBeyondEncoderCapacity() async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try checkpoint.changeJSON("rl_agent_config.json") { $0["max_len"] = 513 }
        await #expect(throws: DecisionError.invalidConfiguration("Agent context exceeds the encoder capacity.")) {
            try await LayaModel.load(from: checkpoint.directory)
        }
    }

    @Test(arguments: ["cls_token", "sep_token", "pad_token", "mask_token"])
    func rejectsMissingTokenizerSpecialToken(key: String) async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try checkpoint.changeJSON("tokenizer/tokenizer_config.json") { $0.removeValue(forKey: key) }
        await #expect(throws: DecisionError.invalidCheckpoint("Tokenizer is missing \(key).")) {
            try await LayaModel.load(from: checkpoint.directory)
        }
    }

    @Test func rejectsTokenizerSpecialTokenOutsideVocabulary() async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try checkpoint.changeJSON("encoder/config.json") { $0["vocab_size"] = 3 }
        await #expect(throws: DecisionError.invalidCheckpoint("Special tokens are outside the encoder vocabulary.")) {
            try await LayaModel.load(from: checkpoint.directory)
        }
    }

    @Test func rejectsMalformedTokenizerJSON() async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try Data("not JSON".utf8).write(to: checkpoint.directory.appending(path: "tokenizer/tokenizer.json"))
        // Tokenizer decoding errors belong to the dependency, so only require a thrown error.
        await #expect(throws: (any Error).self) { try await LayaModel.load(from: checkpoint.directory) }
    }

    @Test func rejectsMissingCheckpointWeights() async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try FileManager.default.removeItem(at: checkpoint.directory.appending(path: "model.safetensors"))
        await #expect(throws: (any Error).self) { try await LayaModel.load(from: checkpoint.directory) }
    }

    @Test(arguments: TemporaryLayaCheckpoint.WeightMutation.allCases)
    func rejectsCheckpointParameterMismatch(mutation: TemporaryLayaCheckpoint.WeightMutation) async throws {
        let checkpoint = try TemporaryLayaCheckpoint()
        defer { checkpoint.remove() }
        try checkpoint.changeWeights(mutation)
        // Keep the safetensors file valid; strict network validation must reject the parameter mismatch.
        await #expect(throws: (any Error).self) { try await LayaModel.load(from: checkpoint.directory) }
    }
}

struct TemporaryLayaCheckpoint {
    let directory: URL

    init() throws {
        directory = Self.nonexistentDirectory()
        try FileManager.default.copyItem(at: LayaParityTests.directory, to: directory)
    }

    static func nonexistentDirectory() -> URL {
        FileManager.default.temporaryDirectory.appending(path: "PicoDecisions-tests-\(UUID().uuidString)")
    }

    func remove() { try? FileManager.default.removeItem(at: directory) }

    func changeJSON(_ path: String, change: (inout [String: Any]) throws -> Void) throws {
        let url = directory.appending(path: path)
        var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        try change(&object)
        try JSONSerialization.data(withJSONObject: object).write(to: url)
    }

    enum WeightMutation: String, CaseIterable, Sendable {
        case wrongShape, missingParameter
    }

    func changeWeights(_ mutation: WeightMutation) throws {
        let url = directory.appending(path: "model.safetensors")
        let data = try Data(contentsOf: url)
        let headerLength = data.prefix(8).enumerated().reduce(UInt64(0)) {
            $0 | (UInt64($1.element) << ($1.offset * 8))
        }
        let payloadStart = 8 + Int(headerLength)
        var header = try #require(JSONSerialization.jsonObject(with: data.subdata(in: 8..<payloadStart)) as? [String: Any])
        switch mutation {
        case .wrongShape:
            let key = "act_head.layers.2.weight"
            var parameter = try #require(header[key] as? [String: Any])
            #expect(parameter["shape"] as? [Int] == [2, 256])
            parameter["shape"] = [1, 512] // Same element count and payload, incompatible head dimensions.
            header[key] = parameter
        case .missingParameter:
            let removedValue = header.removeValue(forKey: "act_head.layers.2.bias")
            let removedParameter = try #require(removedValue as? [String: Any])
            header["unexpected.weight"] = removedParameter
        }
        var encoded = try JSONSerialization.data(withJSONObject: header)
        while !encoded.count.isMultiple(of: 8) { encoded.append(0x20) }
        var rewritten = Data()
        for byte in 0..<8 {
            rewritten.append(UInt8(truncatingIfNeeded: UInt64(encoded.count) >> (byte * 8)))
        }
        rewritten.append(encoded)
        rewritten.append(data.subdata(in: payloadStart..<data.count))
        try rewritten.write(to: url)
    }
}
