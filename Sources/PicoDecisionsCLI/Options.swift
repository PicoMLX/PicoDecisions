import Foundation
import PicoDecisionsMLX

enum CLIError: Error, LocalizedError {
    case usage(String)
    var errorDescription: String? {
        switch self { case .usage(let message): message }
    }
}

struct Options {
    enum Command: String { case demo, benchmark, evaluate }
    let command: Command
    let model: URL
    let precision: LayaPrecision
    let batchSize: Int
    let output: URL?
    let iterations: Int
    let warmup: Int
    let questionCounts: [Int]
    let stateFile: URL?
    let dataset: URL?
    let maximumCandidates: Int

    static let help = """
    Usage: picodecisions <demo|benchmark|evaluate> --model DIRECTORY [options]

      --precision float16|float32  Inference precision (default: float16)
      --batch-size N              GPU question batch size, 1...64 (default: 16)
      --output FILE               Save JSON; otherwise write JSON to stdout

    benchmark options:
      --iterations N              Timed calls per workload (default: 20)
      --warmup N                  Extra untimed calls after first call (default: 3)
      --questions N,N,...         Questions per request (default: 1,8,20)
      --state-file FILE           UTF-8 input; default: a short order request

    evaluate options:
      --dataset FILE             Required labeled candidate-routing JSON dataset
      --max-candidates N          Candidate cap, 1...254 (default: 8)

    Requires local checkpoint files and a Metal-capable Apple silicon Mac.
    No model files are downloaded. Builds can download SwiftPM dependencies.
    Build with Scripts/run.sh; use --help for this text.
    """

    init(_ arguments: [String]) throws {
        guard let first = arguments.first, let command = Command(rawValue: first) else {
            throw CLIError.usage("Choose demo, benchmark, or evaluate.\n\(Self.help)")
        }
        self.command = command
        var allowed: Set<String> = ["--model", "--precision", "--batch-size", "--output"]
        if command == .benchmark { allowed.formUnion(["--iterations", "--warmup", "--questions", "--state-file"]) }
        if command == .evaluate { allowed.formUnion(["--dataset", "--max-candidates"]) }
        var values: [String: String] = [:]
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            guard allowed.contains(key), index + 1 < arguments.count,
                  !arguments[index + 1].hasPrefix("--"), values[key] == nil else {
                throw CLIError.usage("Unknown, duplicate, or incomplete option: \(key)")
            }
            values[key] = arguments[index + 1]
            index += 2
        }
        func integer(_ key: String, default fallback: Int, range: ClosedRange<Int>) throws -> Int {
            guard let value = Int(values[key] ?? String(fallback)), range.contains(value) else {
                throw CLIError.usage("\(key) must be an integer in \(range).")
            }
            return value
        }
        guard let model = values["--model"], !model.isEmpty else {
            throw CLIError.usage("--model DIRECTORY is required.")
        }
        self.model = URL(filePath: model)
        guard let precision = LayaPrecision(rawValue: values["--precision"] ?? "float16") else {
            throw CLIError.usage("--precision must be float16 or float32.")
        }
        self.precision = precision
        batchSize = try integer("--batch-size", default: 16, range: 1...64)
        iterations = try integer("--iterations", default: 20, range: 1...100_000)
        warmup = try integer("--warmup", default: 3, range: 0...10_000)
        maximumCandidates = try integer("--max-candidates", default: 8, range: 1...254)
        let counts = (values["--questions"] ?? "1,8,20").split(separator: ",", omittingEmptySubsequences: false)
        let parsed = counts.compactMap { Int($0) }
        guard parsed.count == counts.count, !parsed.isEmpty,
              parsed.allSatisfy({ (1...256).contains($0) }), Set(parsed).count == parsed.count else {
            throw CLIError.usage("--questions must contain distinct integers in 1...256 separated by commas.")
        }
        questionCounts = parsed
        output = values["--output"].map { URL(filePath: $0) }
        stateFile = values["--state-file"].map { URL(filePath: $0) }
        dataset = values["--dataset"].map { URL(filePath: $0) }
        if command == .evaluate, dataset == nil { throw CLIError.usage("evaluate requires --dataset FILE.") }
        if let output {
            let destination = output.standardizedFileURL.resolvingSymlinksInPath().path
            let checkpointPath = self.model.standardizedFileURL.resolvingSymlinksInPath().path
            let checkpointFiles = ["model.safetensors", "encoder/config.json", "rl_agent_config.json",
                                   "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
                .map { self.model.appending(path: $0) }
            let inputs = ([dataset, stateFile].compactMap { $0 } + checkpointFiles)
                .map { $0.standardizedFileURL.resolvingSymlinksInPath().path }
            guard destination != checkpointPath, !destination.hasPrefix(checkpointPath + "/"),
                  !inputs.contains(destination) else {
                throw CLIError.usage("--output must not overwrite the checkpoint, dataset, or state input.")
            }
        }
    }
}
