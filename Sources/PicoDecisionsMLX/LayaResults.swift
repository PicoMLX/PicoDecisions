// Adapted from Laya and laya-mlx. See NOTICE.
import Foundation
import PicoDecisions

func layaSoftmax(_ logits: [Double]) throws -> [Double] {
    guard let highest = logits.max(), logits.allSatisfy(\.isFinite) else {
        throw DecisionError.nonFiniteOutput
    }
    let values = logits.map { exp($0 - highest) }
    let total = values.reduce(0, +)
    return values.map { $0 / total }
}

func layaResult(_ item: LayaPreparedQuestion, logits: [Float], action: [Float],
                config: LayaAgentConfiguration) throws -> DecisionResult {
    guard logits.count >= item.markers.count, action.count == config.actionCount else {
        throw DecisionError.invalidCheckpoint("Unexpected decision output dimensions.")
    }
    let k = item.markers.count
    let temperature = config.temperature(type: Int(item.type), options: k)
    let p = try layaSoftmax(logits.prefix(k).map { Double($0) / temperature })
    let act = try layaSoftmax(action.map(Double.init))
    let entropy = -p.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
    var confidence = k < 2 ? 1 : min(1, max(0, 1 - entropy / log(Double(k))))
    let answer: DecisionResult.Answer
    switch item.question.kind {
    case .choice(let options):
        // Strict comparison preserves the first option on ties, matching argmax.
        let best = p.indices.dropFirst().reduce(0) { p[$1] > p[$0] ? $1 : $0 }
        answer = .choice(selectedID: options[best].id,
                         probabilities: zip(options, p).map { .init(optionID: $0.0.id, probability: $0.1) })
    case .score:
        let expected = p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
        answer = .score(expectedLevel: expected, probabilities: p)
    case .boolean:
        answer = .boolean(probabilityTrue: p[1])
        confidence = max(p[1], 1 - p[1])
    }
    return DecisionResult(id: item.question.id, answer: answer, confidence: confidence,
                          actProbability: act[0], inputTokenCount: item.ids.count,
                          inputDiagnostics: item.inputDiagnostics)
}
