#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-android-vulkan-pixel}"
INSTALL_DIR="${INSTALL_DIR:-${REPO_ROOT}/pkg-android-vulkan/llama.cpp}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MODEL_WEIGHTS_DIR="${MODEL_WEIGHTS_DIR:-${model_weights:-/home/wuwang/workspace/model_weights}}"
MODEL_PATH="${MODEL_PATH:-${MODEL:-${LLAMA_MODEL:-${GGUF_MODEL:-}}}}"
PROMPT="${PROMPT:-请用一句话回答：你好，我正在验证 Pixel Vulkan GPU 构建。}"
N_PREDICT="${N_PREDICT:-24}"
CTX_SIZE="${CTX_SIZE:-1024}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-31}"
ANDROID_STL="${ANDROID_STL:-c++_shared}"
ANDROID_NDK_ROOT_RESOLVED="${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}"
ADB_BIN="${ADB_BIN:-adb}"
ADB_SERIAL="${ADB_SERIAL:-}"
DEVICE_ROOT="${DEVICE_ROOT:-/data/local/tmp/llama.cpp-vulkan}"
DEVICE_MODEL_DIR="${DEVICE_MODEL_DIR:-/data/local/tmp/gguf}"
NGL="${NGL:-99}"
CHECK_ONLY=0

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [--check-only]

Env vars:
  ANDROID_NDK_ROOT   Android NDK root
  BUILD_DIR          Android build directory
  INSTALL_DIR        cmake --install output directory
  MODEL_PATH         Explicit GGUF model path for verification
  MODEL              Alias of MODEL_PATH
  LLAMA_MODEL        Alias of MODEL_PATH
  GGUF_MODEL         Alias of MODEL_PATH
  MODEL_WEIGHTS_DIR  Directory containing GGUF models
  model_weights      Legacy alias of MODEL_WEIGHTS_DIR
  ADB_SERIAL         Optional adb serial
  DEVICE_ROOT        Install path on the Pixel
  DEVICE_MODEL_DIR   Model directory on the Pixel
  ANDROID_PLATFORM   Android API level, default android-31
  PROMPT             Verification prompt
EOF
}

adb_cmd() {
    if [[ -n "${ADB_SERIAL}" ]]; then
        "${ADB_BIN}" -s "${ADB_SERIAL}" "$@"
    else
        "${ADB_BIN}" "$@"
    fi
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

resolve_device_vulkan_name() {
    local output
    output="$(adb_cmd shell "cd ${DEVICE_ROOT}/llama.cpp && export LD_LIBRARY_PATH=\$PWD/lib && ./bin/llama-cli --list-devices" 2>&1 || true)"
    printf '%s\n' "${output}" > "${LOG_FILE}.devices"

    local device
    device="$(printf '%s\n' "${output}" | sed -n 's/^[[:space:]]*\([^[:space:]:]*Vulkan[^[:space:]:]*\):.*/\1/p; s/^[[:space:]]*[0-9]\+:[[:space:]]\+\([^[:space:]:]*Vulkan[^[:space:]:]*\):.*/\1/p' | head -n 1)"
    if [[ -z "${device}" ]]; then
        echo "No Vulkan runtime device found on Pixel. See ${LOG_FILE}.devices" >&2
        exit 1
    fi
    printf '%s\n' "${device}"
}

find_android_ndk() {
    local candidates=()
    [[ -n "${ANDROID_NDK_ROOT_RESOLVED}" ]] && candidates+=("${ANDROID_NDK_ROOT_RESOLVED}")
    candidates+=(
        "${HOME}/Android/Sdk/ndk-bundle"
        "${HOME}/Android/Sdk/ndk"
        "${HOME}/android-sdk/ndk-bundle"
        "${HOME}/android-sdk/ndk"
        "${HOME}/workspace/llama_for_phone/tools/android-ndk"
        "/opt/android-sdk/ndk"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/build/cmake/android.toolchain.cmake" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
        if [[ -d "${candidate}" ]]; then
            local nested
            nested="$(find "${candidate}" -mindepth 1 -maxdepth 2 -type f -path '*/build/cmake/android.toolchain.cmake' | sort | tail -n 1 || true)"
            if [[ -n "${nested}" ]]; then
                dirname "$(dirname "$(dirname "${nested}")")"
                return 0
            fi
        fi
    done

    local fallback
    fallback="$(find "${HOME}" /opt /usr -name android.toolchain.cmake 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "${fallback}" ]]; then
        dirname "$(dirname "$(dirname "${fallback}")")"
        return 0
    fi

    echo "Unable to locate Android NDK. Set ANDROID_NDK_ROOT." >&2
    exit 1
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

command -v cmake >/dev/null 2>&1 || { echo "Missing cmake" >&2; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "Missing ninja" >&2; exit 1; }
command -v glslc >/dev/null 2>&1 || { echo "Missing glslc" >&2; exit 1; }
command -v "${ADB_BIN}" >/dev/null 2>&1 || { echo "Missing adb: ${ADB_BIN}" >&2; exit 1; }
adb_cmd get-state >/dev/null 2>&1 || {
    echo "adb device is not available. Connect the Pixel or set ADB_SERIAL." >&2
    exit 1
}

MODEL_PATH_RESOLVED="$(resolve_model_path)"
ANDROID_NDK_ROOT_RESOLVED="$(find_android_ndk)"
ANDROID_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT_RESOLVED}/build/cmake/android.toolchain.cmake"
MODEL_BASENAME="$(basename "${MODEL_PATH_RESOLVED}")"
LOG_FILE="${BUILD_DIR}/verify-pixel-vulkan.log"

echo "Build dir    : ${BUILD_DIR}"
echo "Install dir  : ${INSTALL_DIR}"
echo "Android NDK  : ${ANDROID_NDK_ROOT_RESOLVED}"
echo "Model        : ${MODEL_PATH_RESOLVED}"
echo "Device root  : ${DEVICE_ROOT}"
echo

if (( CHECK_ONLY == 1 )); then
    echo "Check-only mode: dependency and model checks passed."
    exit 0
fi

cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN_FILE}" \
    -DANDROID_ABI="${ANDROID_ABI}" \
    -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
    -DANDROID_STL="${ANDROID_STL}" \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_VULKAN=ON \
    -DCMAKE_C_FLAGS="-march=armv8.7-a" \
    -DCMAKE_CXX_FLAGS="-march=armv8.7-a"

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j"$(nproc)"
cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}" --config "${BUILD_TYPE}"
install -m 0644 \
    "${ANDROID_NDK_ROOT_RESOLVED}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
    "${INSTALL_DIR}/lib/libc++_shared.so"

adb_cmd shell "rm -rf ${DEVICE_ROOT} && mkdir -p ${DEVICE_ROOT} ${DEVICE_MODEL_DIR}"
adb_cmd push "${INSTALL_DIR}" "${DEVICE_ROOT}/"
adb_cmd push "${MODEL_PATH_RESOLVED}" "${DEVICE_MODEL_DIR}/${MODEL_BASENAME}"

VULKAN_DEVICE="$(resolve_device_vulkan_name)"
echo "Vulkan dev   : ${VULKAN_DEVICE}"

CLI_ARGS=(
    --model "${DEVICE_MODEL_DIR}/${MODEL_BASENAME}"
    --prompt "${PROMPT}"
    --ctx-size "${CTX_SIZE}"
    --n-predict "${N_PREDICT}"
    --simple-io
    --n-gpu-layers "${NGL}"
    --device "${VULKAN_DEVICE}"
    --no-warmup
    --single-turn
    --no-display-prompt
    --no-mmap
    --parallel-load
    --async-io-load
    --load-micro-stats
    --verbose
)

REMOTE_CMD="cd ${DEVICE_ROOT}/llama.cpp && export LD_LIBRARY_PATH=\$PWD/lib && export TMPDIR=/data/local/tmp && ./bin/llama-cli"
for arg in "${CLI_ARGS[@]}"; do
    REMOTE_CMD+=" $(printf '%q' "${arg}")"
done

set -o pipefail
adb_cmd shell "${REMOTE_CMD}" 2>&1 | tee "${LOG_FILE}"
set +o pipefail

grep -Eiq 'Vulkan|vk' "${LOG_FILE}" || {
    echo "Verification log does not show Vulkan backend activity: ${LOG_FILE}" >&2
    exit 1
}

grep -Eiq 'load_tensors: offload|load_tensors: offloaded|offloading' "${LOG_FILE}" || {
    echo "Verification log does not show model offload activity: ${LOG_FILE}" >&2
    exit 1
}

grep -Fq 'llama_load_all_tensor_data_async_io: loading tensor data with' "${LOG_FILE}" || {
    echo "Verification log does not show async IO loading: ${LOG_FILE}" >&2
    exit 1
}

grep -Fq 'preallocated' "${LOG_FILE}" || {
    echo "Verification log does not show async IO staging buffer preallocation: ${LOG_FILE}" >&2
    exit 1
}

grep -Fq 'llama_print_load_timing_stats: read wall time' "${LOG_FILE}" || {
    echo "Verification log does not show load micro timing stats: ${LOG_FILE}" >&2
    exit 1
}

echo
echo "Pixel Vulkan build and model verification completed."
echo "Verification log: ${LOG_FILE}"
