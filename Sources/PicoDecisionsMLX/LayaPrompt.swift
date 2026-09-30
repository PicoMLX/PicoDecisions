// Adapted from Laya and laya-mlx. See NOTICE.
import Foundation
import PicoDecisions
import Tokenizers

/// Handling of state text that exceeds a checkpoint's context budget.
public enum LayaInputPolicy: Sendable {
    /// Fail so callers can shorten the state explicitly.
    case reject
    /// Match upstream Laya by keeping the beginning of the state that fits.
    case truncateState
}

protocol LayaTokenizing: Sendable {
    var cls: Int { get }
    var sep: Int { get }
    var pad: Int { get }
    var mask: Int { get }
    var maskText: String { get }
    func encode(_ text: String) -> [Int]
}

struct LayaTokenizer: LayaTokenizing {
    let tokenizer: any Tokenizers.Tokenizer
    let cls: Int
    let sep: Int
    let pad: Int
    let mask: Int
    let maskText: String

    init(tokenizer: any Tokenizers.Tokenizer, configuration: Data) throws {
        let config = try JSONSerialization.jsonObject(with: configuration) as? [String: Any] ?? [:]
        func token(_ key: String) throws -> (String, Int) {
            let value = config[key] as? String ?? (config[key] as? [String: Any])?["content"] as? String
            guard let value, let id = tokenizer.convertTokenToId(value) else {
                throw DecisionError.invalidCheckpoint("Tokenizer is missing \(key).")
            }
            return (value, id)
        }
        self.tokenizer = tokenizer
        cls = try token("cls_token").1
        sep = try token("sep_token").1
        pad = try token("pad_token").1
        (maskText, mask) = try token("mask_token")
    }

    func encode(_ text: String) -> [Int] {
        // Rust tokenizers emits no tokens for empty input. Swift's Metaspace
        // pre-tokenizer otherwise inserts a leading-space token for mmBERT.
        text.isEmpty ? [] : tokenizer.encode(text: text, addSpecialTokens: false)
    }
}

struct LayaPreparedQuestion: Sendable {
    let question: DecisionQuestion
    let ids: [Int32]
    let markers: [Int32]
    let type: Int32
}

func prepareLaya(_ request: DecisionRequest, tokenizer tok: any LayaTokenizing,
                 config: LayaAgentConfiguration, vocabularySize: Int,
                 policy: LayaInputPolicy) throws -> [LayaPreparedQuestion] {
    try request.validate()
    if request.questions.isEmpty { return [] }
    let state = tok.encode(request.state.replacingOccurrences(of: tok.maskText, with: " "))
    return try request.questions.map { question in
        try Task.checkCancellation()
        let type: Int32
        let kind: String
        let options: [String]
        switch question.kind {
        case .choice(let values):
            type = 0; kind = "choice"
            options = values.map { $0.description.isEmpty ? $0.id : "\($0.id): \($0.description)" }
        case .score(let values):
            type = 1; kind = "score"
            options = values.enumerated().map { "level \($0.offset): \($0.element)" }
        case .boolean:
            type = 2; kind = "noul"
            let criteria = question.booleanCriteria ?? .init()
            options = ["false: \(criteria.falseDescription)", "true: \(criteria.trueDescription)"]
        }
        guard options.count <= 255 else {
            throw DecisionError.capacityExceeded("Laya supports at most 255 options per question.")
        }
        func clean(_ text: String) -> String { text.replacingOccurrences(of: tok.maskText, with: " ") }
        let instructions = tok.encode("\(kind) question: \(clean(question.instructions))")
        var optionIDs = options.map { [tok.mask] + tok.encode(" " + clean($0)).prefix(48) }
        var budget = config.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let perOption = max(4, (config.headMaxLength - 16) / optionIDs.count)
            optionIDs = optionIDs.map { Array($0.prefix(perOption)) }
            budget = config.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        var ids = [tok.cls] + instructions.prefix(max(8, budget)) + [tok.sep]
        var markers: [Int32] = []
        for option in optionIDs {
            markers.append(Int32(ids.count))
            ids += option
        }
        ids.append(tok.sep)
        let room = config.maxLength - ids.count - 1
        guard room >= 0 else {
            throw DecisionError.capacityExceeded("Question \(question.id) and its options exceed the context budget.")
        }
        if case .reject = policy, state.count > room {
            throw DecisionError.capacityExceeded("State for \(question.id) has \(state.count) tokens; only \(room) fit.")
        }
        ids += state.prefix(room)
        ids.append(tok.sep)
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize && $0 <= Int32.max }) else {
            throw DecisionError.invalidCheckpoint("Tokenizer emitted an ID outside the encoder vocabulary.")
        }
        return LayaPreparedQuestion(question: question, ids: ids.map(Int32.init), markers: markers, type: type)
    }
}
