import Foundation
import Darwin
import Metal
import MLX
import PicoDecisions
import PicoDecisionsMLX

@main
enum PicoDecisionsCommand {
    static let defaultState = "Find my order from yesterday. Please do not cancel it."

    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments.isEmpty || arguments.contains("--help") || arguments.contains("-h") {
            print(Options.help)
            return
        }
        do {
            let options = try Options(arguments)
            let dataset: RoutingDataset?
            let datasetFingerprint: String?
            if let url = options.dataset {
                let data = try Data(contentsOf: url)
                dataset = try JSONDecoder().decode(RoutingDataset.self, from: data)
                datasetFingerprint = CheckpointMetadata.digest(data: data)
                try dataset?.validate(maximumCandidates: options.maximumCandidates)
            } else { dataset = nil; datasetFingerprint = nil }
            guard MTLCreateSystemDefaultDevice() != nil else {
                throw CLIError.usage("Inference requires an available Metal GPU.")
            }
            let runtime = RuntimeMetadata()
            // Fingerprinting reads the files, so load time reflects a warm filesystem cache.
            // It is excluded from inference timings and identifies the exact tested files.
            let checkpoint = try CheckpointMetadata(directory: options.model)
            Memory.peakMemory = 0
            let clock = ContinuousClock()
            let start = clock.now
            let model = try await LayaModel.load(from: options.model, precision: options.precision,
                                                 batchSize: options.batchSize)
            let loadTime = milliseconds(start.duration(to: clock.now))
            switch options.command {
            case .demo:
                let request = demoRequest()
                let results = try await model.predict(request)
                try write(results.map(PredictionReport.init), to: options.output)
            case .benchmark:
                let state = try options.stateFile.map { try String(contentsOf: $0, encoding: .utf8) } ?? defaultState
                let afterLoad = Memory.snapshot()
                var workloads: [BenchmarkWorkload] = []
                for count in options.questionCounts {
                    let request = benchmarkRequest(state: state, questionCount: count)
                    Memory.peakMemory = 0
                    let firstStart = clock.now
                    let first = try await model.predict(request)
                    let firstTime = milliseconds(firstStart.duration(to: clock.now))
                    for _ in 0..<options.warmup { _ = try await model.predict(request) }
                    var samples: [Double] = []
                    for _ in 0..<options.iterations {
                        let sampleStart = clock.now
                        _ = try await model.predict(request)
                        samples.append(milliseconds(sampleStart.duration(to: clock.now)))
                    }
                    let latency = try LatencyStatistics(samples)
                    workloads.append(.init(questionCount: count,
                        inputTokenCounts: first.compactMap(\.inputTokenCount),
                        firstPredictionMilliseconds: firstTime, latency: latency,
                        questionsPerSecond: Double(count) * 1000 / latency.meanMilliseconds,
                        memory: Memory.snapshot()))
                }
                try write(BenchmarkReport(runtime: runtime, checkpoint: checkpoint,
                    precision: options.precision.rawValue, batchSize: options.batchSize,
                    stateCharacterCount: state.count, warmupCalls: options.warmup,
                    loadMilliseconds: loadTime, memoryAfterLoad: afterLoad, workloads: workloads), to: options.output)
            case .evaluate:
                guard let dataset, let datasetFingerprint else {
                    throw CLIError.usage("evaluate requires --dataset FILE.")
                }
                let selector = try ToolDecisionSelector(model: model, maximumCandidates: options.maximumCandidates)
                Memory.peakMemory = 0
                var outcomes: [RoutingOutcome] = []
                for item in dataset.cases {
                    let sampleStart = clock.now
                    let selection = try await selector.select(query: item.state, candidates: item.candidates.map(\.candidate))
                    outcomes.append(.init(id: item.id, expectedToolIDs: item.expectedToolIDs,
                        candidateIDs: item.candidates.map(\.id), retrievalSelectedID: item.candidates.first?.id,
                        selectedID: selection.selectedCandidate?.id,
                        candidateProbabilities: selection.candidateProbabilities.map { .init(id: $0.optionID, probability: $0.probability) },
                        noMatchProbability: selection.noMatchProbability,
                        latencyMilliseconds: milliseconds(sampleStart.duration(to: clock.now)),
                        inputTokenCount: selection.inputTokenCount, confidence: selection.confidence,
                        actProbability: selection.actProbability))
                }
                let snapshot = Memory.snapshot()
                try write(EvaluationReport(runtime: runtime, checkpoint: checkpoint,
                    precision: options.precision.rawValue, batchSize: options.batchSize,
                    datasetName: dataset.name, datasetDescription: dataset.description,
                    datasetSHA256: datasetFingerprint, loadMilliseconds: loadTime,
                    maximumCandidates: options.maximumCandidates, metrics: try RoutingMetrics(outcomes),
                    latency: try LatencyStatistics(outcomes.map(\.latencyMilliseconds)),
                    memory: .init(activeBytes: snapshot.activeMemory, cacheBytes: snapshot.cacheMemory,
                                  peakActiveBytes: snapshot.peakMemory), outcomes: outcomes), to: options.output)
            }
        } catch {
            FileHandle.standardError.write(Data("picodecisions: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    static func write<T: Encodable>(_ value: T, to output: URL?) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        var data = try encoder.encode(value)
        data.append(0x0a)
        if let output { try data.write(to: output, options: .atomic) }
        else { FileHandle.standardOutput.write(data) }
    }

    static func benchmarkRequest(state: String, questionCount: Int) -> DecisionRequest {
        .init(state: state, questions: (0..<questionCount).map { index in
            .init(id: "route_\(index)", instructions: "Which action does the user request?",
                kind: .choice(options: [.init(id: "lookup", description: "Find an existing order"),
                    .init(id: "cancel", description: "Cancel an existing order"),
                    .init(id: "none", description: "Neither action is requested")]))
        })
    }

    static func demoRequest() -> DecisionRequest {
        let choice = benchmarkRequest(state: defaultState, questionCount: 1).questions[0]
        return .init(state: defaultState, questions: [choice,
            .init(id: "urgency", instructions: "How urgent is this request?", kind: .score(levels: ["routine", "soon", "immediate"])),
            .init(id: "cancel", instructions: "Does the user request cancellation?", kind: .boolean)])
    }
}
