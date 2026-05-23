#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${PWD}/scripts/build_vulkan_pixel_verify.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

make_fake_ndk() {
    local ndk_dir=$1
    mkdir -p "${ndk_dir}/build/cmake" "${ndk_dir}/toolchains/llvm/prebuilt/linux-x86_64/lib/linux/aarch64"
    : > "${ndk_dir}/build/cmake/android.toolchain.cmake"
}

make_fake_cmake_that_must_not_run() {
    mkdir -p "${TMP_DIR}/bin"
    cat > "${TMP_DIR}/bin/cmake" <<'EOF_CMAKE'
#!/usr/bin/env bash
echo "cmake should not be invoked during --check-only" >&2
exit 42
EOF_CMAKE
    chmod +x "${TMP_DIR}/bin/cmake"
}

run_check_only_with_clean_env() {
    env -i \
        PATH="${TMP_DIR}/bin:/usr/bin:/bin" \
        HOME="${TMP_DIR}/home" \
        "$@" \
        bash "${SCRIPT}" --check-only
}

make_fake_cmake_that_must_not_run

explicit_ndk="${TMP_DIR}/explicit-ndk"
make_fake_ndk "${explicit_ndk}"
output=$(run_check_only_with_clean_env ANDROID_NDK_ROOT="${explicit_ndk}")
grep -F "${explicit_ndk}" <<< "${output}" >/dev/null

sdk_dir="${TMP_DIR}/sdk"
older_ndk="${sdk_dir}/ndk/27.1.12297006"
newer_ndk="${sdk_dir}/ndk/27.3.13750724"
make_fake_ndk "${older_ndk}"
make_fake_ndk "${newer_ndk}"
output=$(run_check_only_with_clean_env ANDROID_HOME="${sdk_dir}")
grep -F "${newer_ndk}" <<< "${output}" >/dev/null
