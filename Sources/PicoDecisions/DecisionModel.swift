/// A model that answers typed questions without generating text.
///
/// Implementations own tokenization, model execution, calibration, and capacity checks.
/// Return one result per question, in request order, preserving question identifiers.
/// Invalid requests and unsupported inputs must throw rather than silently lose options.
public protocol DecisionModel: Sendable {
    func predict(_ request: DecisionRequest) async throws -> [DecisionResult]
}

/// Textual state and ordered questions to evaluate against it.
public struct DecisionRequest: Sendable, Equatable {
    public let state: String
    public let questions: [DecisionQuestion]

    public init(state: String, questions: [DecisionQuestion]) {
        self.state = state
        self.questions = questions
    }
}

public struct DecisionQuestion: Sendable, Equatable, Identifiable {
    public let id: String
    public let instructions: String
    public let kind: Kind

    /// Choice options and score levels retain their supplied order.
    public enum Kind: Sendable, Equatable {
        case choice(options: [DecisionOption])
        case score(levels: [String])
        case boolean
    }

    public init(id: String, instructions: String, kind: Kind) {
        self.id = id
        self.instructions = instructions
        self.kind = kind
    }
}

public struct DecisionOption: Sendable, Equatable, Identifiable {
    public let id: String
    public let description: String

    public init(id: String, description: String) {
        self.id = id
        self.description = description
    }
}

public struct DecisionResult: Sendable, Equatable, Identifiable {
    /// The identifier of the corresponding question.
    public let id: String
    public let answer: Answer
    /// Model-specific confidence; this is not a guarantee of correctness.
    public let confidence: Double?
    /// Probability of acting rather than escalating, when provided by the model.
    public let actProbability: Double?
    /// Tokens consumed by this question, including its prompt and state.
    public let inputTokenCount: Int?

    public enum Answer: Sendable, Equatable {
        case choice(selectedID: String, probabilities: [OptionProbability])
        /// Probabilities follow the request's level order; expectedLevel is zero-based.
        case score(expectedLevel: Double, probabilities: [Double])
        case boolean(probabilityTrue: Double)
    }

    public init(id: String, answer: Answer, confidence: Double? = nil,
                actProbability: Double? = nil, inputTokenCount: Int? = nil) {
        self.id = id
        self.answer = answer
        self.confidence = confidence
        self.actProbability = actProbability
        self.inputTokenCount = inputTokenCount
    }
}

public struct OptionProbability: Sendable, Equatable {
    public let optionID: String
    public let probability: Double

    public init(optionID: String, probability: Double) {
        self.optionID = optionID
        self.probability = probability
    }
}
