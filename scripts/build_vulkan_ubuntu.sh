#!/usr/bin/env bash
set -euo pipefail

# One-click helper for Ubuntu/Debian:
# 1) checks missing build/Vulkan dependencies
# 2) installs missing packages
# 3) configures and builds llama.cpp with Vulkan backend enabled

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-vulkan}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CHECK_ONLY=0

for arg in "$@"; do
    case "${arg}" in
        --check-only)
            CHECK_ONLY=1
            ;;
        --help|-h)
            cat <<'EOF'
Usage:
  scripts/build_vulkan_ubuntu.sh [--check-only]

Env vars:
  BUILD_DIR   Build output directory (default: ./build-vulkan)
  BUILD_TYPE  CMake build type (default: Release)
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 1
            ;;
    esac
done

if [[ ! -f /etc/os-release ]]; then
    echo "Cannot detect Linux distribution (/etc/os-release missing)." >&2
    exit 1
fi

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" && "${ID_LIKE:-}" != *"debian"* ]]; then
    echo "This script currently supports Ubuntu/Debian only." >&2
    echo "Detected: ${PRETTY_NAME:-unknown}" >&2
    exit 1
fi

missing_items=()
missing_packages=()

add_package_if_missing() {
    local pkg="$1"
    for existing in "${missing_packages[@]:-}"; do
        if [[ "${existing}" == "${pkg}" ]]; then
            return 0
        fi
    done
    missing_packages+=("${pkg}")
}

if ! command -v git >/dev/null 2>&1; then
    missing_items+=("git")
    add_package_if_missing "git"
fi

if ! command -v cmake >/dev/null 2>&1; then
    missing_items+=("cmake")
    add_package_if_missing "cmake"
fi

if ! command -v ninja >/dev/null 2>&1; then
    missing_items+=("ninja")
    add_package_if_missing "ninja-build"
fi

if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
    missing_items+=("gcc/g++/make toolchain")
    add_package_if_missing "build-essential"
fi

if ! command -v pkg-config >/dev/null 2>&1; then
    missing_items+=("pkg-config")
    add_package_if_missing "pkg-config"
fi

if ! command -v ccache >/dev/null 2>&1; then
    missing_items+=("ccache")
    add_package_if_missing "ccache"
fi

if ! command -v glslc >/dev/null 2>&1; then
    missing_items+=("glslc")
    add_package_if_missing "glslc"
fi

if ! command -v vulkaninfo >/dev/null 2>&1; then
    missing_items+=("vulkaninfo")
    add_package_if_missing "vulkan-tools"
fi

if [[ ! -f /usr/include/vulkan/vulkan.h ]]; then
    missing_items+=("Vulkan headers")
    add_package_if_missing "libvulkan-dev"
fi

echo "Detected system: ${PRETTY_NAME:-unknown}"
echo "Build dir      : ${BUILD_DIR}"
echo "Build type     : ${BUILD_TYPE}"
echo

if ((${#missing_items[@]} > 0)); then
    echo "Missing tools/deps:"
    printf '  - %s\n' "${missing_items[@]}"
    echo
    echo "Will install packages:"
    printf '  - %s\n' "${missing_packages[@]}"
else
    echo "All required packages are already installed."
fi

if ((CHECK_ONLY == 1)); then
    echo
    echo "Check-only mode: skipping install and build."
    exit 0
fi

if ((${#missing_packages[@]} > 0)); then
    echo
    echo "Installing missing packages..."
    sudo apt-get update
    sudo apt-get install -y "${missing_packages[@]}"
fi

echo
echo "Tool versions:"
cmake --version | head -n 1
ninja --version
gcc --version | head -n 1
g++ --version | head -n 1
pkg-config --version
glslc --version | head -n 1 || true
vulkaninfo --summary >/dev/null 2>&1 || echo "Warning: vulkaninfo runtime probe failed (compile can still succeed)."

echo
echo "Configuring CMake (Vulkan ON)..."
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DGGML_VULKAN=ON \
    -DGGML_CCACHE=ON

echo "Building..."
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j"$(nproc)"

echo
echo "Build complete."
echo "Try:"
echo "  ${BUILD_DIR}/bin/llama-cli -m /path/to/model.gguf -p 'hello' -ngl 99"
