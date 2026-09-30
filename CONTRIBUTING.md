# Contributing

PicoDecisions is an experimental Swift package. Keep the backend-independent API
in `PicoDecisions` and MLX-specific behavior in `PicoDecisionsMLX`. The public API
is provisional; describe any API changes in your pull request.

## Local validation

The package targets macOS 15+ and iOS 18+, and requires Swift 6.3+ because the pinned
MLX Swift dependency uses a Swift 6.3 manifest. For development, use Xcode 26.6+
on a macOS version supported by Xcode. The inference suite needs an Apple silicon
Mac with an available Metal GPU and a full Xcode installation. Select the intended Xcode
with `DEVELOPER_DIR` if multiple versions are installed. If Xcode reports a missing
Metal compiler, install its component with
`xcodebuild -downloadComponent MetalToolchain`.

```sh
# Core API and CLI unit tests; no GPU required.
Scripts/test.sh core

# Complete suite, including the checked-in deterministic synthetic checkpoint.
Scripts/test.sh inference

# Complete suite plus trained-model parity; download weights as in README first.
PICODECISIONS_LAYA_MODEL="$PWD/models/laya-multilingual" Scripts/test.sh checkpoint
```

The script uses the versions in `Package.resolved`. Initial dependency resolution
requires network access; tests never download model weights. `inference` always
uses the synthetic fixture, and `checkpoint` explicitly enables the trained-model
tests. Use the Xcode-backed script for repeatable inference validation and an
explicit check that a Metal GPU is available.

Pass extra arguments to the underlying test tool after the mode, for example
`Scripts/test.sh core --skip-build`. To reuse build
directories, set `PICODECISIONS_SWIFT_SCRATCH_PATH` for SwiftPM, or
`PICODECISIONS_DERIVED_DATA_PATH` and `PICODECISIONS_PACKAGE_CACHE_PATH` for Xcode.
Xcode's default build directory is the ignored `DerivedData/` folder.

## GitHub Actions

The **Core tests** workflow runs automatically on pushes and pull requests using
a standard hosted macOS 26 arm64 runner and Xcode 26.6 (Swift 6.3.3). It validates
the core API and CLI logic, and compiles the package's executable and test targets.
This does not establish MLX inference parity.

The toolchain selection follows the official
[macOS 26 runner image](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-arm64-Readme.md)
and [Swift 6.3.3 release announcement](https://forums.swift.org/t/announcing-swift-6-3-3/87888).

The **Metal inference tests** workflow is manually dispatched by a maintainer.
Register a self-hosted Apple silicon Mac with the labels `macOS`, `ARM64`, and
`picodecisions-metal`, install Xcode with its Metal compiler, and confirm
`Scripts/test.sh inference` passes on it. Dispatch the workflow for the reviewed
branch when changing model loading, tokenization, calibration, or neural operations.
It checks for an actual Metal device and runs the complete synthetic suite.
Failed Xcode result bundles are retained for seven days. Trained-model parity
remains an explicit local check.

GitHub documents standard macOS arm64 jobs as virtual machines and advertises GPU
hardware acceleration for its paid macOS XLarge runners. This configuration uses
a physical self-hosted Mac for inference, following
[MLX Swift's own test workflow](https://github.com/ml-explore/mlx-swift/blob/0.31.6/.github/workflows/pull_request.yml).
See [standard runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [larger runner specifications](https://docs.github.com/en/actions/reference/runners/larger-runners).

## Pull requests and reproducibility

The **iOS library builds** workflow compiles device and simulator libraries and
Metal resources on hosted runners. Locally, use `Scripts/build-ios.sh device`
and `Scripts/build-ios.sh simulator`. These checks need no device or signing and
do not run inference. See [iOS validation](Docs/IOSValidation.md) for SDK and cache
details.

Explain the behavior being changed, the relevant checks run, and any validation
that remains outstanding. Add focused regression coverage for correctness changes.
Inference changes should retain Python parity within the documented tolerances.
Separate implementation parity from decision accuracy and performance claims;
include the checkpoint revision, precision, hardware, and workload for measurements.

Keep trained weights, downloaded models, build outputs, and private datasets out
of commits. The tiny test weights are deterministic random data. Fixture provenance
and regeneration commands are recorded in [Docs/Validation.md](Docs/Validation.md).
Preserve attribution notices when adapting upstream code or reference fixtures.

Contributions are licensed under the repository's [Apache-2.0 license](LICENSE).
See [NOTICE](NOTICE) for upstream acknowledgments.
