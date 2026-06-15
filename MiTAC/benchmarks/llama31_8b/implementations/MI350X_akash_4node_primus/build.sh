#!/usr/bin/env bash
# build.sh — Build the per-arch MLPerf Training v6.0 image (one ROCm arch).
#
# The dual-arch image (single Dockerfile, gfx942+gfx950) was the prime suspect
# for MI325X FP8 NaN at first eval. This wrapper splits the build per arch.
#
# Usage:
#   bash build.sh --config MI325X
#   bash build.sh --config MI350X
#
# Flags:
#   --config <MI325X|MI350X>   selects Dockerfile.<config> and the image tag.
#   --bg                       run the build in the background (nohup).
#   --no-version               skip the auto VERSION suffix on the image tag,
#                              even when a VERSION file is present at the
#                              archive root (default: auto-suffix when VERSION
#                              is found).
#   -h | --help                print usage.
#
# Environment variables:
#   IMAGE_TAG                  override the default image tag.
#                              default: rocm/amd-mlperf:llama3.1_8b_v6.0_<config>
#   BUILD_LOG                  path to write the build log.
#                              default: $HOME/build_<config>_<timestamp>.log
#   DOCKER_BUILD_ARGS          extra args appended to `docker build`.
#                              e.g. "--no-cache --pull"
#
# Exit codes:
#   0  build succeeded (or backgrounded successfully)
#   1  Dockerfile not found / docker daemon error
#   2  invalid arguments

set -euo pipefail

# -- arg parsing ------------------------------------------------------------

CONFIG=""
RUN_BG=0
NO_VERSION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)     CONFIG="${2:-}"; shift 2 ;;
    --config=*)   CONFIG="${1#*=}"; shift ;;
    --bg)         RUN_BG=1; shift ;;
    --no-version) NO_VERSION=1; shift ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Try: bash $0 --help" >&2
      exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: --config is required (MI325X or MI350X)" >&2
  echo "Try: bash $0 --help" >&2
  exit 2
fi

case "$CONFIG" in
  MI325X|MI350X) ;;
  *)
    echo "Error: unknown --config '$CONFIG'. Valid: MI325X, MI350X" >&2
    exit 2
    ;;
esac

# -- resolve paths ----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DOCKERFILE="Dockerfile.${CONFIG}"
if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Error: $DOCKERFILE not found in $SCRIPT_DIR" >&2
  exit 1
fi

# Auto-detect repo VERSION marker (written by compress.sh at archive root, four
# levels above this script). When present (and --no-version not set), namespace
# the image tag so multiple unpacked versions on the same host don't clobber
# each other's images. --no-version forces the unversioned tag even if VERSION
# exists (useful for sharing one image across multiple unpacked trees).
REPO_VERSION_FILE="${SCRIPT_DIR}/../../../../VERSION"
REPO_VERSION=""
if [[ "$NO_VERSION" -eq 0 && -f "$REPO_VERSION_FILE" ]]; then
  REPO_VERSION="$(head -n1 "$REPO_VERSION_FILE" | tr -d '[:space:]')"
fi

# default tag follows the config (and VERSION if present)
if [[ -n "$REPO_VERSION" ]]; then
  IMAGE_TAG="${IMAGE_TAG:-rocm/amd-mlperf:llama3.1_8b_v6.0_${CONFIG}_${REPO_VERSION}}"
else
  IMAGE_TAG="${IMAGE_TAG:-rocm/amd-mlperf:llama3.1_8b_v6.0_${CONFIG}}"
fi

# default log path follows the config + timestamp
TS=$(date +%Y%m%d_%H%M%S)
BUILD_LOG="${BUILD_LOG:-$HOME/build_${CONFIG}_${TS}.log}"

mkdir -p "$(dirname "$BUILD_LOG")"

# -- print plan -------------------------------------------------------------

cat <<EOF
=================== build.sh plan ===================
config:        $CONFIG
dockerfile:    $DOCKERFILE
image tag:     $IMAGE_TAG
repo version:  ${REPO_VERSION:-$([[ $NO_VERSION -eq 1 ]] && echo "(suppressed by --no-version)" || echo "(none — git checkout / no VERSION file)")}
log file:      $BUILD_LOG
extra args:    ${DOCKER_BUILD_ARGS:-(none)}
background:    $([[ $RUN_BG -eq 1 ]] && echo yes || echo no)
=====================================================
EOF

# -- preflight: docker reachable -------------------------------------------

if ! docker info >/dev/null 2>&1; then
  echo "Error: docker daemon is not reachable. Run 'docker info' to debug." >&2
  exit 1
fi

# -- build ------------------------------------------------------------------

echo "Starting build at $(date '+%Y-%m-%d %H:%M:%S')..."

# shellcheck disable=SC2086
if [[ "$RUN_BG" -eq 1 ]]; then
  nohup docker build \
    -t "$IMAGE_TAG" \
    -f "$DOCKERFILE" \
    ${DOCKER_BUILD_ARGS:-} \
    . > "$BUILD_LOG" 2>&1 < /dev/null &
  BUILD_PID=$!
  echo "Build running in background. PID=$BUILD_PID"
  echo "Monitor:  tail -f $BUILD_LOG"
  echo "Verify:   docker images $IMAGE_TAG"
else
  docker build \
    -t "$IMAGE_TAG" \
    -f "$DOCKERFILE" \
    ${DOCKER_BUILD_ARGS:-} \
    . 2>&1 | tee "$BUILD_LOG"
  echo ""
  echo "Build done at $(date '+%Y-%m-%d %H:%M:%S')."
  echo "Image:    $IMAGE_TAG"
  echo "Log:      $BUILD_LOG"
fi
