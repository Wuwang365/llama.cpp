#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECK_ONLY=0
CMAKE_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--check-only" ]]; then
        CHECK_ONLY=1
    else
        CMAKE_ARGS+=("${arg}")
    fi
done

is_android_ndk() {
    [[ -f "$1/build/cmake/android.toolchain.cmake" ]]
}

discover_android_ndk() {
    local explicit="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${ANDROID_NDK:-}}}"
    if [[ -n "${explicit}" ]]; then
        if is_android_ndk "${explicit}"; then
            printf "%s\n" "${explicit}"
            return 0
        fi
        echo "Configured Android NDK path is not valid: ${explicit}" >&2
        return 1
    fi

    local sdk
    for sdk in "${ANDROID_HOME:-}" "${ANDROID_SDK_ROOT:-}" "${HOME}/Android/Sdk"; do
        if [[ -n "${sdk}" && -d "${sdk}/ndk" ]]; then
            while IFS= read -r candidate; do
                if is_android_ndk "${candidate}"; then
                    printf "%s\n" "${candidate}"
                    return 0
                fi
            done < <(find "${sdk}/ndk" -mindepth 1 -maxdepth 1 -type d | sort -Vr)
        fi
    done

    local candidate
    for candidate in /opt/android-ndk* "${ROOT_DIR}/../android-ndk"* "${ROOT_DIR}/../../android-ndk"*; do
        if [[ -d "${candidate}" && -f "${candidate}/build/cmake/android.toolchain.cmake" ]]; then
            printf "%s\n" "${candidate}"
            return 0
        fi
    done

    return 1
}

if ! ANDROID_NDK_HOME="$(discover_android_ndk)"; then
    echo "ANDROID_NDK_HOME, ANDROID_NDK_ROOT, or ANDROID_NDK must point to an Android NDK" >&2
    echo "Install the NDK under \$ANDROID_HOME/ndk or export one of those variables." >&2
    exit 1
fi

ABI="${ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-31}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android-vulkan-${ABI}}"
VULKAN_HPP_INCLUDE="${VULKAN_HPP_INCLUDE:-}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "Android NDK check OK: ${ANDROID_NDK_HOME}"
    echo "Android ABI: ${ABI}"
    echo "Android platform: ${ANDROID_PLATFORM}"
    exit 0
fi

if [[ -z "${VULKAN_HPP_INCLUDE}" ]]; then
    for candidate in \
        "${ROOT_DIR}/../third_party/Vulkan-Headers/include" \
        "${ROOT_DIR}/../../llama_for_phone/third_party/Vulkan-Headers/include" \
        "/usr/include"; do
        if [[ -f "${candidate}/vulkan/vulkan.hpp" ]]; then
            VULKAN_HPP_INCLUDE="${candidate}"
            break
        fi
    done
fi

EXTRA_CXX_FLAGS="${CMAKE_CXX_FLAGS:-}"
if [[ -n "${VULKAN_HPP_INCLUDE}" ]]; then
    EXTRA_CXX_FLAGS="${EXTRA_CXX_FLAGS} -I${VULKAN_HPP_INCLUDE}"
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${ABI}" \
    -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -DGGML_VULKAN=ON \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    "${CMAKE_ARGS[@]}"

cmake --build "${BUILD_DIR}" --target \
    llama-cli \
    llama-completion \
    llama-server \
    llama-switch-bench

if [[ -z "${ADB_SERIAL:-}" ]]; then
    echo "Android Vulkan build complete. Set ADB_SERIAL and MODEL or REMOTE_MODEL to run the optional Pixel smoke test." >&2
    exit 0
fi

REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/llama-vulkan-verify}"
REMOTE_MODEL="${REMOTE_MODEL:-/data/local/tmp/gguf/Qwen3-1.7B-Q8_0.gguf}"
ADB=(adb -s "${ADB_SERIAL}")

"${ADB[@]}" shell "mkdir -p '${REMOTE_DIR}'"
"${ADB[@]}" push \
    "${BUILD_DIR}/bin/libggml-base.so" \
    "${BUILD_DIR}/bin/libggml-cpu.so" \
    "${BUILD_DIR}/bin/libggml-vulkan.so" \
    "${BUILD_DIR}/bin/libggml.so" \
    "${BUILD_DIR}/bin/libllama.so" \
    "${BUILD_DIR}/bin/libmtmd.so" \
    "${BUILD_DIR}/bin/llama-completion" \
    "${BUILD_DIR}/bin/llama-server" \
    "${BUILD_DIR}/bin/llama-switch-bench" \
    "${REMOTE_DIR}/"

LIBOMP="$(find "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt" -path "*/lib/linux/aarch64/libomp.so" | head -1)"
if [[ -n "${LIBOMP}" ]]; then
    "${ADB[@]}" push "${LIBOMP}" "${REMOTE_DIR}/"
fi

LOCAL_MODEL="${MODEL_PATH:-${MODEL:-}}"
if [[ -n "${LOCAL_MODEL}" ]]; then
    "${ADB[@]}" push "${LOCAL_MODEL}" "${REMOTE_DIR}/model.gguf"
    REMOTE_MODEL="${REMOTE_DIR}/model.gguf"
fi

"${ADB[@]}" shell "cd '${REMOTE_DIR}' && \
    chmod +x ./llama-completion ./llama-server ./llama-switch-bench && \
    LD_LIBRARY_PATH=. \
    GGML_VK_PREFER_HOST_MEMORY=1 \
    LLAMA_VK_HOST_VISIBLE_DIRECT_SET=1 \
    LLAMA_ASYNC_IO_QUEUES='${LLAMA_ASYNC_IO_QUEUES:-2}' \
    LLAMA_WEIGHT_BARRIER_TARGET='${LLAMA_WEIGHT_BARRIER_TARGET:-8}' \
    LLAMA_WEIGHT_UNLOAD_KEEP_REGEX='${LLAMA_WEIGHT_UNLOAD_KEEP_REGEX:-(^token_embd\\.weight$|^blk\\.[0-3]\\.)}' \
    ./llama-completion -m '${REMOTE_MODEL}' -p '${PROMPT:-Hello}' -n '${N_PREDICT:-1}' -c '${CTX_SIZE:-4096}' -ngl '${N_GPU_LAYERS:-99}' \
        --no-mmap --parallel-load --async-io-load --load-micro-stats \
        --unload-all-after-load --no-warmup"
