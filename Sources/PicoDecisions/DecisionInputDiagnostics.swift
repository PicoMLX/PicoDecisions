/// Token counts before and after a model fits each component into its prompt.
///
/// Counts describe tokenized, normalized prompt text, including any labels the
/// model adds to instructions and options. Structural separator and marker
/// tokens are excluded. They need not equal the token count of the raw text.
public struct DecisionInputDiagnostics: Sendable, Equatable, Codable {
    public struct TokenCounts: Sendable, Equatable, Codable {
        public let originalTokenCount: Int
        public let retainedTokenCount: Int
        public var wasTruncated: Bool { retainedTokenCount < originalTokenCount }

        public init(originalTokenCount: Int, retainedTokenCount: Int) {
            self.originalTokenCount = originalTokenCount
            self.retainedTokenCount = retainedTokenCount
        }
    }

    public struct Option: Sendable, Equatable, Codable {
        /// Choice ID, zero-based score index as a string, or "false"/"true".
        public let optionID: String
        public let tokens: TokenCounts

        public init(optionID: String, tokens: TokenCounts) {
            self.optionID = optionID
            self.tokens = tokens
        }
    }

    public let state: TokenCounts
    public let instructions: TokenCounts
    /// Retains the request's option order, including an explicit no-match option.
    public let options: [Option]
    public var wasTruncated: Bool {
        state.wasTruncated || instructions.wasTruncated || options.contains { $0.tokens.wasTruncated }
    }

    public init(state: TokenCounts, instructions: TokenCounts, options: [Option]) {
        self.state = state
        self.instructions = instructions
        self.options = options
    }

    private enum CodingKeys: String, CodingKey {
        case state, instructions, options, wasTruncated
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(state: try values.decode(TokenCounts.self, forKey: .state),
                  instructions: try values.decode(TokenCounts.self, forKey: .instructions),
                  options: try values.decode([Option].self, forKey: .options))
    }

    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(state, forKey: .state)
        try values.encode(instructions, forKey: .instructions)
        try values.encode(options, forKey: .options)
        try values.encode(wasTruncated, forKey: .wasTruncated)
    }
}
