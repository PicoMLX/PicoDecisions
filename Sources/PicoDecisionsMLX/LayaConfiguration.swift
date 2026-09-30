import Foundation
import PicoDecisions

struct LayaEncoderConfiguration: Decodable, Sendable {
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let normEps: Float
    let normBias: Bool
    let attentionBias: Bool
    let mlpBias: Bool
    let localAttention: Int
    let maxPositionEmbeddings: Int
    let layerTypes: [String]
    let ropeBases: [Float]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size", numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads", normEps = "norm_eps"
        case normBias = "norm_bias", attentionBias = "attention_bias", mlpBias = "mlp_bias"
        case localAttention = "local_attention", maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types", ropeParameters = "rope_parameters"
        case globalEvery = "global_attn_every_n_layers", globalTheta = "global_rope_theta"
        case localTheta = "local_rope_theta", modelType = "model_type", activation = "hidden_activation"
    }

    private struct RopeParameters: Decodable {
        let rope_type: String?
        let rope_theta: Float?
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        localAttention = try c.decodeIfPresent(Int.self, forKey: .localAttention) ?? 128
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        let kind = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "modernbert"
        let activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "gelu"
        let every = try c.decodeIfPresent(Int.self, forKey: .globalEvery) ?? 3
        guard kind == "modernbert", activation == "gelu",
              vocabSize > 0, hiddenSize > 0, intermediateSize > 0,
              numHiddenLayers > 0, numAttentionHeads > 0, every > 0,
              hiddenSize.isMultiple(of: numAttentionHeads),
              (hiddenSize / numAttentionHeads).isMultiple(of: 2),
              hiddenSize.isMultiple(of: max(1, hiddenSize / 64)),
              localAttention >= 0, maxPositionEmbeddings > 0,
              normEps.isFinite, normEps > 0 else {
            throw DecisionError.invalidConfiguration("Unsupported ModernBERT dimensions, activation, or normalization.")
        }
        let types = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0..<numHiddenLayers).map { $0 % every == 0 ? "full_attention" : "sliding_attention" }
        guard types.count == numHiddenLayers,
              types.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) }) else {
            throw DecisionError.invalidConfiguration("Unsupported encoder layer types.")
        }
        let rope = try c.decodeIfPresent([String: RopeParameters].self, forKey: .ropeParameters) ?? [:]
        let global = try c.decodeIfPresent(Float.self, forKey: .globalTheta) ?? 160_000
        let local = try c.decodeIfPresent(Float.self, forKey: .localTheta) ?? 10_000
        ropeBases = try types.map { type in
            let parameters = rope[type]
            let base = parameters?.rope_theta ?? (type == "full_attention" ? global : local)
            guard (parameters?.rope_type ?? "default") == "default", base.isFinite, base > 0 else {
                throw DecisionError.invalidConfiguration("Only unscaled ModernBERT RoPE is supported.")
            }
            return base
        }
        layerTypes = types
    }
}

struct LayaAgentConfiguration: Decodable, Sendable {
    let encoder: String
    let headLayers: Int
    let maxLength: Int
    let headMaxLength: Int
    let actionCount: Int
    let temperature: [Double]
    let temperatureByOptions: [String: Double]

    enum CodingKeys: String, CodingKey {
        case encoder, headLayers = "head_layers", maxLength = "max_len"
        case headMaxLength = "head_max_len", actionCosts = "act_costs"
        case temperature, temperatureByOptions = "temperature_by_options"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoder = try c.decode(String.self, forKey: .encoder)
        headLayers = try c.decode(Int.self, forKey: .headLayers)
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? 512
        headMaxLength = try c.decodeIfPresent(Int.self, forKey: .headMaxLength) ?? 192
        actionCount = (try c.decodeIfPresent([String: Double].self, forKey: .actionCosts) ?? [:]).count + 1
        temperature = try c.decodeIfPresent([Double].self, forKey: .temperature) ?? [1, 1, 1]
        temperatureByOptions = try c.decodeIfPresent([String: Double].self, forKey: .temperatureByOptions) ?? [:]
        guard headLayers >= 0, headMaxLength > 4, headMaxLength < maxLength,
              temperature.count == 3,
              (temperature + Array(temperatureByOptions.values)).allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw DecisionError.invalidConfiguration("Invalid decision head, token budget, or calibration.")
        }
    }

    func temperature(type: Int, options: Int) -> Double {
        let count = options <= 2 ? "2" : options <= 5 ? "3-5" : options <= 10 ? "6-10" : "11+"
        return max(1e-3, temperatureByOptions["\(["choice", "score", "noul"][type]):\(count)"] ?? temperature[type])
    }
}
