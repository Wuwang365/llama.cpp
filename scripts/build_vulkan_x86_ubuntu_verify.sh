#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-vulkan-ubuntu}"
MODEL="${MODEL:-${BUILD_DIR}/tinyllamas/stories15M-q4_0.gguf}"

"${ROOT_DIR}/scripts/build_vulkan_ubuntu.sh"

ctest --test-dir "${BUILD_DIR}" -R 'test-weight-buffer|test-weight-unload|test-parallel-load' --output-on-failure

GGML_VK_PREFER_HOST_MEMORY="${GGML_VK_PREFER_HOST_MEMORY:-1}" \
LLAMA_VK_HOST_VISIBLE_DIRECT_SET="${LLAMA_VK_HOST_VISIBLE_DIRECT_SET:-1}" \
LLAMA_WEIGHT_UNLOAD_KEEP_REGEX="${LLAMA_WEIGHT_UNLOAD_KEEP_REGEX:-(^token_embd\\.weight$|^blk\\.[0-3]\\.)}" \
timeout "${VERIFY_TIMEOUT:-60s}" \
"${BUILD_DIR}/bin/llama-completion" \
    -m "${MODEL}" \
    -p "${PROMPT:-Hello}" \
    -n "${N_PREDICT:-1}" \
    --no-mmap \
    --parallel-load \
    --async-io-load \
    --load-micro-stats \
    --unload-all-after-load \
    --no-warmup

"${BUILD_DIR}/bin/llama-switch-bench" \
    -m "${MODEL}" \
    -p "${PROMPT:-Hello}" \
    --runs "${RUNS:-5}" \
    --unload-layer-window "${UNLOAD_LAYER_WINDOW:-4:5}" \
    --unload-output \
    --no-mmap \
    --parallel-load \
    --async-io-load \
    --load-micro-stats \
    --output-format "${OUTPUT_FORMAT:-csv}"
