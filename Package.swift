// swift-tools-version: 6.3

import PackageDescription

let package = Package(
    name: "PicoDecisions",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        .library(name: "PicoDecisions", targets: ["PicoDecisions"]),
        .library(name: "PicoDecisionsMLX", targets: ["PicoDecisionsMLX"]),
        .executable(name: "picodecisions", targets: ["PicoDecisionsCLI"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.6")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.4")
    ],
    targets: [
        .target(name: "PicoDecisions"),
        .target(name: "PicoDecisionsMLX", dependencies: [
            "PicoDecisions",
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "Tokenizers", package: "swift-transformers")
        ]),
        .executableTarget(name: "PicoDecisionsCLI", dependencies: [
            "PicoDecisions", "PicoDecisionsMLX",
            .product(name: "MLX", package: "mlx-swift")
        ]),
        .testTarget(name: "PicoDecisionsCLITests", dependencies: ["PicoDecisionsCLI"]),
        .testTarget(name: "PicoDecisionsTests", dependencies: ["PicoDecisions"]),
        .testTarget(name: "PicoDecisionsMLXTests", dependencies: ["PicoDecisionsMLX"],
                    resources: [.copy("Fixtures")])
    ]
)
