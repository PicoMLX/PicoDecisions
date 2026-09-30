import Foundation

public enum DecisionError: Error, Sendable, Equatable, LocalizedError {
    case invalidRequest(String)
    case invalidConfiguration(String)
    case invalidCheckpoint(String)
    case capacityExceeded(String)
    case nonFiniteOutput

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let message): "Invalid decision request: \(message)"
        case .invalidConfiguration(let message): "Invalid model configuration: \(message)"
        case .invalidCheckpoint(let message): "Invalid checkpoint: \(message)"
        case .capacityExceeded(let message): "Decision capacity exceeded: \(message)"
        case .nonFiniteOutput: "The model produced non-finite output. Try FP32 inference."
        }
    }
}

extension DecisionRequest {
    /// Validate identifiers and question structure before inference.
    public func validate() throws {
        var ids = Set<String>()
        for question in questions {
            guard !question.id.isEmpty, ids.insert(question.id).inserted else {
                throw DecisionError.invalidRequest("Question identifiers must be nonempty and unique.")
            }
            guard !question.instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw DecisionError.invalidRequest("Question \(question.id) needs instructions.")
            }
            switch question.kind {
            case .choice(let options):
                guard !options.isEmpty,
                      options.allSatisfy({ !$0.id.isEmpty }),
                      Set(options.map(\.id)).count == options.count else {
                    throw DecisionError.invalidRequest("Choice options must have nonempty, unique identifiers.")
                }
            case .score(let levels):
                guard !levels.isEmpty else {
                    throw DecisionError.invalidRequest("A score question needs at least one level.")
                }
            case .boolean: break
            }
        }
    }
}
