# iOS compile validation

The iOS check compiles the `PicoDecisionsMLX` library scheme and its dependencies,
including `PicoDecisions` and MLX's Metal resources. It targets arm64 with an
iOS 18 deployment minimum for both the device and simulator SDKs. The package's
Swift 6.3 minimum follows the pinned MLX Swift dependency.

A successful build checks source and SDK compatibility. It does not run inference,
load checkpoints, verify app integration, or establish physical-device correctness,
latency, or memory requirements. Physical iOS inference remains unvalidated.

## Local commands

Use a full Xcode installation with Swift 6.3+, the iOS SDKs, and its Metal compiler.
Select the intended installation with `DEVELOPER_DIR` if multiple versions are
installed. Xcode 26.6 is the CI toolchain; newer supported toolchains can be used
locally.

```sh
Scripts/build-ios.sh device
Scripts/build-ios.sh simulator
```

These commands use generic destinations. Neither needs a connected device,
provisioning profile, simulator boot, simulator runtime download, nor access to a
Metal GPU. They compile in Debug configuration with signing, indexing, and code
coverage disabled, and use `dwarf` debug information to avoid separate dSYM output.
The CLI and test targets are outside this library build.

The script restricts dependency resolution to `Package.resolved`. An initial
checkout requires network access to fetch those dependencies; it never downloads
model weights. Build outputs default to the ignored `DerivedData/ios-device` and
`DerivedData/ios-simulator` directories. Set `PICODECISIONS_DERIVED_DATA_PATH` to
override the exact output directory, `PICODECISIONS_PACKAGE_CACHE_PATH` to reuse
Xcode source checkouts, and `PICODECISIONS_BUILD_JOBS` to change compiler parallelism
from the default of two. Use separate output directories for the two SDKs.

If Xcode reports a missing Metal compiler, install its component with:

```sh
xcodebuild -downloadComponent MetalToolchain
```

Apple documents this command in
[downloading and installing additional Xcode components](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components).
The local script leaves compiler discovery to Xcode and does not install components.

Xcode may also ask to trust dependency build plugins on a fresh checkout. CI
explicitly sets `PICODECISIONS_TRUST_PACKAGE_PLUGINS=1` for the reviewed, resolved
dependencies. This adds command-scoped `-skipPackagePluginValidation`; it does not
change Xcode's global settings. The pinned MLX `CudaBuild` plugin returns no build
commands on Apple platforms. Local builds keep Xcode's plugin validation enabled
unless the caller explicitly sets that variable after reviewing the plugins.

## GitHub Actions

The **iOS library builds** workflow runs on pushes and pull requests. Its device
and simulator matrix uses hosted `macos-26` arm64 runners with Xcode 26.6
(Swift 6.3.3). The selected SDKs are supplied by that Xcode installation; the
deployment target stays at iOS 18. See the official
[runner image inventory](https://github.com/actions/runner-images/blob/main/images/macos/macos-26-arm64-Readme.md)
and [Swift 6.3.3 announcement](https://forums.swift.org/t/announcing-swift-6-3-3/87888).

CI checks `xcodebuild -showComponent MetalToolchain`, downloads the component only
when Xcode reports it absent, and verifies installation before building. It checks
the reported installation status because the query's exit code alone does not
establish availability. Unknown query results fail the job. Source checkouts are
cached by the resolved dependency graph; compiled outputs remain separate per SDK.
Failed build logs are retained for seven days. No self-hosted runner is needed.

## Runtime validation still required

On September 30, 2026, both script modes completed successfully with Xcode 27
and its iOS 27 SDKs on the development M1 Max, targeting arm64 and iOS 18. The
builds included the current core API and MLX library plus Metal resources.
The hosted Xcode 26.6 matrix still needs its first run.

The next runtime check should load the small synthetic checkpoint in an iOS app
or test host on a physical device, compare FP32 and FP16 outputs with the existing
reference, and exercise cancellation and repeated calls. A separate trained-model
run should measure memory, loading, and completed-inference latency on representative
devices. Existing macOS parity and performance results do not establish these iOS
properties.
