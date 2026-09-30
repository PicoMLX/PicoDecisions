import CryptoKit
import Foundation
import Metal
import MLX
import PicoDecisions

struct RuntimeMetadata: Encodable {
    let operatingSystem = ProcessInfo.processInfo.operatingSystemVersionString
    let gpu = MTLCreateSystemDefaultDevice()?.name ?? "unavailable"
    let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    let processorCount = ProcessInfo.processInfo.processorCount
    let buildConfiguration: String = {
        #if DEBUG
        "debug"
        #else
        "release"
        #endif
    }()
}

struct CheckpointMetadata: Encodable {
    let directoryName: String
    let weightsSHA256: String
    let encoderConfigurationSHA256: String
    let agentConfigurationSHA256: String
    let tokenizerSHA256: String
    let tokenizerConfigurationSHA256: String

    init(directory: URL) throws {
        directoryName = directory.lastPathComponent
        weightsSHA256 = try Self.digest(directory.appending(path: "model.safetensors"))
        encoderConfigurationSHA256 = try Self.digest(directory.appending(path: "encoder/config.json"))
        agentConfigurationSHA256 = try Self.digest(directory.appending(path: "rl_agent_config.json"))
        tokenizerSHA256 = try Self.digest(directory.appending(path: "tokenizer/tokenizer.json"))
        tokenizerConfigurationSHA256 = try Self.digest(directory.appending(path: "tokenizer/tokenizer_config.json"))
    }

    static func digest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let data = try file.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func digest(data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

struct LatencyStatistics: Encodable {
    let samplesMilliseconds: [Double]
    let minimumMilliseconds: Double
    let medianMilliseconds: Double
    let p95Milliseconds: Double
    let meanMilliseconds: Double

    init(_ samples: [Double]) throws {
        guard !samples.isEmpty, samples.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw CLIError.usage("Latency samples must be nonempty, finite, and nonnegative.")
        }
        samplesMilliseconds = samples
        let ordered = samples.sorted()
        minimumMilliseconds = ordered[0]
        let middle = ordered.count / 2
        medianMilliseconds = ordered.count.isMultiple(of: 2)
            ? (ordered[middle - 1] + ordered[middle]) / 2 : ordered[middle]
        // Nearest-rank percentile: small runs retain their highest observed latency.
        p95Milliseconds = ordered[max(0, Int(ceil(Double(ordered.count) * 0.95)) - 1)]
        meanMilliseconds = samples.reduce(0, +) / Double(samples.count)
    }
}

func milliseconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
}

struct PredictionReport: Encodable {
    struct Probability: Encodable { let id: String; let probability: Double }
    let id: String
    let kind: String
    let selectedID: String?
    let probabilities: [Probability]?
    let expectedLevel: Double?
    let probabilityTrue: Double?
    let confidence: Double?
    let actProbability: Double?
    let inputTokenCount: Int?
    let inputDiagnostics: DecisionInputDiagnostics?

    init(_ result: DecisionResult) {
        id = result.id
        confidence = result.confidence
        actProbability = result.actProbability
        inputTokenCount = result.inputTokenCount
        inputDiagnostics = result.inputDiagnostics
        switch result.answer {
        case .choice(let selected, let values):
            kind = "choice"; selectedID = selected; expectedLevel = nil; probabilityTrue = nil
            probabilities = values.map { .init(id: $0.optionID, probability: $0.probability) }
        case .score(let value, let distribution):
            kind = "score"; selectedID = nil; expectedLevel = value; probabilityTrue = nil
            probabilities = distribution.enumerated().map { .init(id: String($0.offset), probability: $0.element) }
        case .boolean(let value):
            kind = "boolean"; selectedID = nil; expectedLevel = nil; probabilityTrue = value
            probabilities = nil
        }
    }
}

struct BenchmarkWorkload: Encodable {
    let questionCount: Int
    let inputTokenCounts: [Int]
    let inputDiagnostics: [DecisionInputDiagnostics]?
    let firstPredictionMilliseconds: Double
    let latency: LatencyStatistics
    let questionsPerSecond: Double
    let memory: Memory.Snapshot
}

struct BenchmarkReport: Encodable {
    let schemaVersion = 1
    let timestamp = Date()
    let runtime: RuntimeMetadata
    let checkpoint: CheckpointMetadata
    let precision: String
    let batchSize: Int
    let stateCharacterCount: Int
    let warmupCalls: Int
    let loadMilliseconds: Double
    let memoryAfterLoad: Memory.Snapshot
    let workloads: [BenchmarkWorkload]
}
