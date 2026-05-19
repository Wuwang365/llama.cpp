#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-vulkan-ubuntu}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_VULKAN=ON \
    -DLLAMA_BUILD_TESTS=ON \
    "$@"

cmake --build "${BUILD_DIR}" --target \
    llama-cli \
    llama-completion \
    llama-server \
    llama-switch-bench \
    test-weight-buffer \
    test-weight-unload \
    test-parallel-load
