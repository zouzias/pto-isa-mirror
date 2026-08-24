#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

repo_dir=""

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
}

main() {
  require_command git
  require_command cmake
  require_command make

  local prefix="${1:-${A5_GTEST_PREFIX:-${HOME}/.local/gtest}}"
  repo_dir="$(mktemp -d "${TMPDIR:-/tmp}/googletest.XXXXXX")"
  local build_dir="${repo_dir}/build"

  trap 'rm -rf "${repo_dir}"' EXIT

  echo "Installing PIC-enabled googletest into ${prefix}"
  git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git "${repo_dir}"

  cmake -S "${repo_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_GMOCK=OFF \
    -DCMAKE_INSTALL_PREFIX="${prefix}"

  cmake --build "${build_dir}" --parallel "$(nproc)"
  cmake --install "${build_dir}"

  echo "Installed GTest config under ${prefix}"
}

main "$@"
