#!/bin/bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: Scripts/test.sh [core|inference|checkpoint] [test-tool arguments...]

  core        Run core API and CLI unit tests with SwiftPM; no GPU required.
  inference   Run the complete synthetic suite with Xcode and Metal (default).
  checkpoint  Also run parity tests against PICODECISIONS_LAYA_MODEL.

Requires macOS 15+ and Swift 6.3+. Inference needs Apple silicon, a Metal GPU,
and a full Xcode installation with its Metal compiler.

Optional paths:
  PICODECISIONS_SWIFT_SCRATCH_PATH   SwiftPM build directory for core tests
  PICODECISIONS_DERIVED_DATA_PATH    Xcode build directory (default: DerivedData)
  PICODECISIONS_PACKAGE_CACHE_PATH   Xcode source-package cache
USAGE
}

mode="${1:-inference}"
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
    -h|--help) usage; exit 0 ;;
    core|inference|checkpoint) ;;
    *) usage >&2; exit 2 ;;
esac

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"

xcrun swift --version
if [ "$mode" = core ]; then
    swift_args=(test --force-resolved-versions --filter '^PicoDecisions(Tests|CLITests)[.]')
    if [ -n "${PICODECISIONS_SWIFT_SCRATCH_PATH:-}" ]; then
        swift_args+=(--scratch-path "$PICODECISIONS_SWIFT_SCRATCH_PATH")
    fi
    exec xcrun swift "${swift_args[@]}" "$@"
fi

if [ "$(uname -m)" != arm64 ]; then
    echo "Inference tests require an Apple silicon Mac running natively." >&2
    exit 1
fi
xcodebuild -version
# Let Xcode locate its Metal compiler; xcrun can miss a mounted MetalToolchain.

derived_data="${PICODECISIONS_DERIVED_DATA_PATH:-$repository_dir/DerivedData}"
mkdir -p "$derived_data/PreflightModuleCache"
xcrun swift -module-cache-path "$derived_data/PreflightModuleCache" -e '
import Foundation
import Metal
guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("Inference tests require an available Metal GPU.\n", stderr)
    exit(1)
}
print("Metal device: \(device.name)")
'

xcode_args=(test -scheme PicoDecisions-Package
    -destination 'platform=macOS,arch=arm64'
    -derivedDataPath "$derived_data"
    -onlyUsePackageVersionsFromResolvedFile
    -parallel-testing-enabled NO)
if [ -n "${PICODECISIONS_PACKAGE_CACHE_PATH:-}" ]; then
    xcode_args+=(-clonedSourcePackagesDirPath "$PICODECISIONS_PACKAGE_CACHE_PATH")
fi

if [ "$mode" = checkpoint ]; then
    if [ -z "${PICODECISIONS_LAYA_MODEL:-}" ] || [ ! -d "$PICODECISIONS_LAYA_MODEL" ]; then
        echo "Set PICODECISIONS_LAYA_MODEL to the local multilingual checkpoint directory." >&2
        exit 2
    fi
    checkpoint_dir="$(cd "$PICODECISIONS_LAYA_MODEL" && pwd)"
    for file in model.safetensors encoder/config.json rl_agent_config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json; do
        if [ ! -f "$checkpoint_dir/$file" ]; then
            echo "Checkpoint is missing $file" >&2
            exit 2
        fi
    done
    export TEST_RUNNER_PICODECISIONS_LAYA_MODEL="$checkpoint_dir"
else
    unset PICODECISIONS_LAYA_MODEL TEST_RUNNER_PICODECISIONS_LAYA_MODEL
fi
exec xcodebuild "${xcode_args[@]}" "$@"
