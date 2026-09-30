import Foundation
import MLX
import MLXNN
import PicoDecisions
import Tokenizers

public enum LayaPrecision: String, Sendable {
    case float16, float32

    var dtype: DType { self == .float16 ? .float16 : .float32 }
}

/// Local-checkpoint Laya inference. Each instance serializes its model access.
/// Model weights and native arrays never cross the actor boundary.
public actor LayaModel: DecisionModel {
    public nonisolated let precision: LayaPrecision
    public nonisolated let batchSize: Int
    public nonisolated let inputPolicy: LayaInputPolicy
    private var network: LayaNetwork?
    private var tokenizer: LayaTokenizer?
    private var encoderConfig: LayaEncoderConfiguration?
    private var agentConfig: LayaAgentConfiguration?

    private init(precision: LayaPrecision, batchSize: Int, inputPolicy: LayaInputPolicy) {
        self.precision = precision
        self.batchSize = batchSize
        self.inputPolicy = inputPolicy
    }

    /// Load a downloaded Laya or laya-mlx checkpoint. No network access is performed.
    /// Both original PyTorch parameter names and converted MLX names are accepted.
    public static func load(from directory: URL, precision: LayaPrecision = .float16,
                            batchSize: Int = 16, inputPolicy: LayaInputPolicy = .reject) async throws -> LayaModel {
        guard batchSize > 0, batchSize <= 64 else {
            throw DecisionError.invalidConfiguration("Batch size must be between 1 and 64.")
        }
        let model = LayaModel(precision: precision, batchSize: batchSize, inputPolicy: inputPolicy)
        try await model.loadCheckpoint(directory)
        return model
    }

    private func loadCheckpoint(_ directory: URL) async throws {
        guard directory.isFileURL else {
            throw DecisionError.invalidCheckpoint("Supply a local checkpoint directory.")
        }
        try Task.checkCancellation()
        let decoder = JSONDecoder()
        let encoder = try decoder.decode(LayaEncoderConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "encoder/config.json")))
        let agent = try decoder.decode(LayaAgentConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "rl_agent_config.json")))
        guard agent.maxLength <= encoder.maxPositionEmbeddings else {
            throw DecisionError.invalidConfiguration("Agent context exceeds the encoder capacity.")
        }
        let tokenizerDirectory = directory.appending(path: "tokenizer")
        let tokenConfig = try Data(contentsOf: tokenizerDirectory.appending(path: "tokenizer_config.json"))
        let backend = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        let tokenizer = try LayaTokenizer(tokenizer: backend, configuration: tokenConfig)
        guard [tokenizer.cls, tokenizer.sep, tokenizer.pad, tokenizer.mask].allSatisfy({
            $0 >= 0 && $0 < encoder.vocabSize && $0 <= Int32.max
        }) else {
            throw DecisionError.invalidCheckpoint("Special tokens are outside the encoder vocabulary.")
        }
        try Task.checkCancellation()
        let model = try Stream.withNewDefaultStream {
            let model = LayaNetwork(encoder: encoder, agent: agent)
            let weights = try Self.sanitize(loadArrays(url: directory.appending(path: "model.safetensors")))
            try model.update(parameters: ModuleParameters.unflattened(weights.mapValues { $0.asType(precision.dtype) }),
                             verify: .all)
            model.train(false)
            eval(model)
            return model
        }
        network = model
        self.tokenizer = tokenizer
        encoderConfig = encoder
        agentConfig = agent
    }

    static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, tensor) in weights {
            var name = key.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
                .replacingOccurrences(of: ".in_proj_bias", with: ".in_proj.bias")
            for prefix in ["scorer", "act_head"] where name.hasPrefix(prefix + ".") && !name.hasPrefix(prefix + ".layers.") {
                name = prefix + ".layers." + name.dropFirst(prefix.count + 1)
            }
            guard result.updateValue(tensor, forKey: name) == nil else {
                throw DecisionError.invalidCheckpoint("Duplicate parameter after conversion: \(name)")
            }
        }
        return result
    }

    public func predict(_ request: DecisionRequest) async throws -> [DecisionResult] {
        try Task.checkCancellation()
        guard let network, let tokenizer, let encoderConfig, let agentConfig else {
            throw DecisionError.invalidCheckpoint("Model is not loaded.")
        }
        let items = try prepareLaya(request, tokenizer: tokenizer, config: agentConfig,
                                    vocabularySize: encoderConfig.vocabSize, policy: inputPolicy)
        var results: [DecisionResult] = []
        for start in stride(from: 0, to: items.count, by: batchSize) {
            try Task.checkCancellation()
            let chunk = Array(items[start..<min(start + batchSize, items.count)])
            let batchResults = try Stream.withNewDefaultStream {
                let tensors = LayaBatch(chunk, pad: Int32(tokenizer.pad))
                let (logits, action) = network(ids: tensors.ids, mask: tensors.mask,
                    positions: tensors.positions, markerMask: tensors.markerMask, types: tensors.types)
                eval(logits, action)
                let scores = logits.asArray(Float.self)
                let actions = action.asArray(Float.self)
                guard scores.allSatisfy(\.isFinite), actions.allSatisfy(\.isFinite) else {
                    throw DecisionError.nonFiniteOutput
                }
                return try chunk.enumerated().map { row, item in
                    try layaResult(item,
                        logits: Array(scores[(row * tensors.options)..<((row + 1) * tensors.options)]),
                        action: Array(actions[(row * agentConfig.actionCount)..<((row + 1) * agentConfig.actionCount)]),
                        config: agentConfig)
                }
            }
            results += batchResults
        }
        try Task.checkCancellation()
        return results
    }
}

struct LayaBatch {
    let ids: MLXArray
    let mask: MLXArray
    let positions: MLXArray
    let markerMask: MLXArray
    let types: MLXArray
    let options: Int

    init(_ items: [LayaPreparedQuestion], pad: Int32) {
        let count = items.count
        let length = items.map(\.ids.count).max() ?? 0
        options = max(2, items.map(\.markers.count).max() ?? 0)
        var ids = [Int32](repeating: pad, count: count * length)
        var mask = [Int32](repeating: 0, count: count * length)
        var positions = [Int32](repeating: 0, count: count * options)
        var markers = [Int32](repeating: 0, count: count * options)
        for (row, item) in items.enumerated() {
            for (column, id) in item.ids.enumerated() {
                ids[row * length + column] = id
                mask[row * length + column] = 1
            }
            for (column, position) in item.markers.enumerated() {
                positions[row * options + column] = position
                markers[row * options + column] = 1
            }
        }
        self.ids = MLXArray(ids).reshaped(count, length)
        self.mask = MLXArray(mask).reshaped(count, length).asType(.bool)
        self.positions = MLXArray(positions).reshaped(count, options)
        self.markerMask = MLXArray(markers).reshaped(count, options).asType(.bool)
        self.types = MLXArray(items.map(\.type))
    }
}
