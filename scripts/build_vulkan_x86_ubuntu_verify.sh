#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-vulkan-x86}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-${model_weights:-/home/wuwang/workspace/model_weights}}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-${LLAMA_MODEL:-${GGUF_MODEL:-}}}}"
PROMPT="${PROMPT:-请用一句话回答：你好，我正在验证 Ubuntu x86 Vulkan 构建。}"
N_PREDICT="${N_PREDICT:-32}"
CTX_SIZE="${CTX_SIZE:-1024}"
THREADS="${THREADS:-$(nproc)}"
NGL="${NGL:-99}"
GGML_CCACHE="${GGML_CCACHE:-OFF}"
CHECK_ONLY=0

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [--check-only]

Env vars:
  BUILD_DIR          Build output directory
  BUILD_TYPE         CMake build type, default Release
  MODEL_PATH         Explicit GGUF model path for verification
  MODEL              Alias of MODEL_PATH
  LLAMA_MODEL        Alias of MODEL_PATH
  GGUF_MODEL         Alias of MODEL_PATH
  MODEL_WEIGHTS_DIR  Directory containing GGUF models
  model_weights      Legacy alias of MODEL_WEIGHTS_DIR
  PROMPT             Verification prompt
  N_PREDICT          Tokens to generate during verification
  CTX_SIZE           Context size used during verification
  THREADS            Build and runtime thread count
  NGL                Number of layers to offload, default 99
  GGML_CCACHE        Enable ccache for the build, default OFF
EOF
}

resolve_model_path() {
    if [[ -n "${MODEL_PATH}" ]]; then
        [[ -f "${MODEL_PATH}" ]] || {
            echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
            exit 1
        }
        printf '%s\n' "${MODEL_PATH}"
        return 0
    fi

    [[ -d "${MODEL_WEIGHTS_DIR}" ]] || {
        echo "model_weights directory does not exist: ${MODEL_WEIGHTS_DIR}" >&2
        exit 1
    }

    local first_model
    first_model="$(find "${MODEL_WEIGHTS_DIR}" -type f -name '*.gguf' | sort | head -n 1)"
    [[ -n "${first_model}" ]] || {
        echo "No .gguf model found under ${MODEL_WEIGHTS_DIR}" >&2
        exit 1
    }
    printf '%s\n' "${first_model}"
}

resolve_vulkan_device() {
    local output
    output="$("${LLAMA_CLI}" --list-devices 2>&1 || true)"
    printf '%s\n' "${output}" > "${BUILD_DIR}/list-devices.log"

    local device
    device="$(printf '%s\n' "${output}" | sed -n 's/^[[:space:]]*[0-9]\+:[[:space:]]\+\([^[:space:]].*Vulkan[^[:space:]]*\)$/\1/p' | head -n 1)"
    printf '%s\n' "${device}"
}

for arg in "$@"; do
    case "${arg}" in
        --check-only)
            CHECK_ONLY=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -f /etc/os-release ]] || {
    echo "Cannot detect Linux distribution (/etc/os-release missing)." >&2
    exit 1
}

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *"debian"* ]]; then
    echo "This script currently supports Ubuntu/Debian only." >&2
    echo "Detected: ${PRETTY_NAME:-unknown}" >&2
    exit 1
fi

REQUIRED_CMDS=(cmake ninja gcc g++ glslc vulkaninfo)
for cmd in "${REQUIRED_CMDS[@]}"; do
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "Missing required command: ${cmd}" >&2
        exit 1
    }
done

[[ -f /usr/include/vulkan/vulkan.h ]] || {
    echo "Missing Vulkan headers at /usr/include/vulkan/vulkan.h" >&2
    exit 1
}

MODEL_PATH_RESOLVED="$(resolve_model_path)"
LLAMA_CLI="${BUILD_DIR}/bin/llama-cli"

echo "System      : ${PRETTY_NAME:-unknown}"
echo "Build dir   : ${BUILD_DIR}"
echo "Build type  : ${BUILD_TYPE}"
echo "Model       : ${MODEL_PATH_RESOLVED}"
echo "Prompt      : ${PROMPT}"
echo "GGML_CCACHE : ${GGML_CCACHE}"
echo

if (( CHECK_ONLY == 1 )); then
    echo "Check-only mode: dependency and model checks passed."
    exit 0
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_VULKAN=ON \
    -DGGML_CCACHE="${GGML_CCACHE}"

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j"${THREADS}"

[[ -x "${LLAMA_CLI}" ]] || {
    echo "llama-cli not found after build: ${LLAMA_CLI}" >&2
    exit 1
}

LOG_FILE="${BUILD_DIR}/verify-vulkan.log"
CLI_ARGS=(
    --model "${MODEL_PATH_RESOLVED}"
    --prompt "${PROMPT}"
    --ctx-size "${CTX_SIZE}"
    --threads "${THREADS}"
    --n-predict "${N_PREDICT}"
    --simple-io
    --no-warmup
)

VULKAN_DEVICE="$(resolve_vulkan_device)"
if [[ -n "${VULKAN_DEVICE}" ]]; then
    echo "Vulkan dev  : ${VULKAN_DEVICE}"
    CLI_ARGS+=(--n-gpu-layers "${NGL}" --device "${VULKAN_DEVICE}")
    VERIFY_EXPECTS_VULKAN=1
else
    echo "Vulkan dev  : not available, falling back to CPU verification"
    CLI_ARGS+=(--n-gpu-layers 0)
    VERIFY_EXPECTS_VULKAN=0
fi

set -o pipefail
"${LLAMA_CLI}" "${CLI_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
set +o pipefail

if (( VERIFY_EXPECTS_VULKAN == 1 )); then
    grep -Eiq 'Vulkan|vk' "${LOG_FILE}" || {
        echo "Verification log does not show Vulkan backend activity: ${LOG_FILE}" >&2
        exit 1
    }

    grep -Eiq 'load_tensors: offload|load_tensors: offloaded|offloading' "${LOG_FILE}" || {
        echo "Verification log does not show model offload activity: ${LOG_FILE}" >&2
        exit 1
    }
fi

echo
echo "Ubuntu x86 build and model verification completed."
echo "Verification log: ${LOG_FILE}"
