#!/bin/bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"
derived_data="${PICODECISIONS_DERIVED_DATA_PATH:-$repository_dir/DerivedData}"
configuration="${PICODECISIONS_BUILD_CONFIGURATION:-Release}"
build_jobs="${PICODECISIONS_BUILD_JOBS:-2}"
case "$configuration" in Debug|Release) ;; *) echo "Use Debug or Release configuration." >&2; exit 2 ;; esac
case "$build_jobs" in ''|*[!0-9]*|0) echo "PICODECISIONS_BUILD_JOBS must be a positive integer." >&2; exit 2 ;; esac
args=(build -scheme picodecisions -configuration "$configuration"
    -destination 'platform=macOS,arch=arm64' -derivedDataPath "$derived_data"
    -jobs "$build_jobs" -onlyUsePackageVersionsFromResolvedFile
    CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO DEBUG_INFORMATION_FORMAT=dwarf
    ENABLE_CODE_COVERAGE=NO CLANG_ENABLE_CODE_COVERAGE=NO CLANG_COVERAGE_MAPPING=NO)
if [ -n "${PICODECISIONS_PACKAGE_CACHE_PATH:-}" ]; then
    args+=(-clonedSourcePackagesDirPath "$PICODECISIONS_PACKAGE_CACHE_PATH")
fi
# Keep compiler output on stderr so stdout remains a machine-readable JSON report.
xcodebuild "${args[@]}" >&2
exec "$derived_data/Build/Products/$configuration/picodecisions" "$@"
