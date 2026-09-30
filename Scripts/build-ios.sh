#!/bin/bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: Scripts/build-ios.sh device|simulator

Compile PicoDecisionsMLX and its dependencies for generic arm64 iOS destinations.
Uses Debug configuration, iOS 18 deployment, and Package.resolved versions.
No device, provisioning profile, simulator boot, or Metal GPU is required.
Requires Swift 6.3+ and Xcode with the iOS SDKs and Metal compiler.

Optional paths and limits:
  PICODECISIONS_DERIVED_DATA_PATH   Xcode build directory
                                  (default: DerivedData/ios-device or ios-simulator)
  PICODECISIONS_PACKAGE_CACHE_PATH Xcode source-package cache
  PICODECISIONS_BUILD_JOBS         Compiler parallelism (default: 2)
  PICODECISIONS_TRUST_PACKAGE_PLUGINS Set to 1 after reviewing resolved plugins
USAGE
}

if [ "$#" -ne 1 ]; then usage >&2; exit 2; fi
platform="$1"
case "$platform" in
    -h|--help) usage; exit 0 ;;
    device) sdk=iphoneos; destination='generic/platform=iOS' ;;
    simulator) sdk=iphonesimulator; destination='generic/platform=iOS Simulator' ;;
    *) usage >&2; exit 2 ;;
esac

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
derived_data="${PICODECISIONS_DERIVED_DATA_PATH:-$repository_dir/DerivedData/ios-$platform}"
build_jobs="${PICODECISIONS_BUILD_JOBS:-2}"
if [[ ! "$build_jobs" =~ ^[0-9]+$ ]] || [[ "$build_jobs" =~ ^0+$ ]]; then
    echo "PICODECISIONS_BUILD_JOBS must be a positive integer." >&2
    exit 2
fi

xcodebuild -version
xcrun swift --version
xcrun --sdk "$sdk" --show-sdk-version

args=(build -scheme PicoDecisionsMLX -configuration Debug
    -sdk "$sdk" -destination "$destination"
    -derivedDataPath "$derived_data" -jobs "$build_jobs"
    -onlyUsePackageVersionsFromResolvedFile
    ARCHS=arm64 ONLY_ACTIVE_ARCH=NO IPHONEOS_DEPLOYMENT_TARGET=18.0
    CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO
    COMPILER_INDEX_STORE_ENABLE=NO DEBUG_INFORMATION_FORMAT=dwarf
    ENABLE_CODE_COVERAGE=NO CLANG_ENABLE_CODE_COVERAGE=NO CLANG_COVERAGE_MAPPING=NO)
if [ -n "${PICODECISIONS_PACKAGE_CACHE_PATH:-}" ]; then
    args+=(-clonedSourcePackagesDirPath "$PICODECISIONS_PACKAGE_CACHE_PATH")
fi
if [ "${PICODECISIONS_TRUST_PACKAGE_PLUGINS:-0}" = 1 ]; then
    args+=(-skipPackagePluginValidation)
fi
# Xcode locates the installed Metal compiler; this compile check never needs a GPU.
exec xcodebuild "${args[@]}"
