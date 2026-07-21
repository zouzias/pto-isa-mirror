#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Appended to the auto-generated postinst by cann-cmake gen_postinst_prerm.py for the rpm/deb package.
#
# IMPORTANT: cann-cmake inlines this whole file into the RPM post section via
# CPACK_RPM_POST_INSTALL_SCRIPT_FILE, so RPM spec macro processor will expand any literal percent.
# The DEB backend uses the same text verbatim. To stay safe on both backends, this script MUST NOT
# contain any literal percent sign (no format strings, no date format, no heredocs with it). Plain shell
# globbing and find -exec are fine.
#
# pto-isa is a header-only library: no wheel/so to install, and no pto-specific install artifacts are
# created at install time. The top-level symlinks (bin/lib64/include/...) and the package database
# (var/ascend_package_db.info) are already handled by the cann-cmake-generated postinst body driven by
# the EngineeringCommon block in pto_isa.xml. RPM/DEB users uninstall via rpm -e / dpkg -r, so no extra
# cann_uninstall.sh entry or ascend_install.info record is created here.
#
# What this script does do: align the installed file/dir permissions with the makeself run package's
# "install for all" mode (IS_FOR_ALL=y, i.e. root install), so that the installed headers are readable
# by every user on the host. The run package's pto_install.sh does this dynamically; rpm/deb packages
# have fixed permissions baked in at pack time, so we replay the same mode bits here.
#
#   builtin files (headers/scripts): 555   (r-xr-xr-x, group/other get r-x, group write stripped)
#   version/scene info files:       444   (r--r--r--, read-only for everyone)
#   package db file:                644   (rw-r--r--, owner writable)
#   directories:                    755 (top-level + custom) or 555 (builtin subtrees)

sourcedir="${INSTALL_PATH}"
PTO_PLATFORM_DIR="pto_isa"
INSTALL_ROOT="${sourcedir}"

# Only run when the install root actually exists (safety check).
if [ -d "${INSTALL_ROOT}" ]; then
    # 1. Builtin headers and scripts: align to 555 (run package BUILTIN_PERM for IS_FOR_ALL=y).
    #    Covers aarch64-linux/include/** and share/info/pto_isa/script/** (excluding version.info).
    #    Use find so we do not depend on find's -xdev; plain -type f is enough.
    find "${INSTALL_ROOT}/aarch64-linux/include" -type f 2>/dev/null -exec chmod 555 {} +
    find "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/script" -type f 2>/dev/null -exec chmod 555 {} +

    # 2. Read-only info files (version header, scene.info, version.info): align to 444.
    [ -f "${INSTALL_ROOT}/aarch64-linux/include/version/pto_isa_version.h" ] && chmod 444 "${INSTALL_ROOT}/aarch64-linux/include/version/pto_isa_version.h" 2>/dev/null || true
    [ -f "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/scene.info" ] && chmod 444 "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/scene.info" 2>/dev/null || true
    [ -f "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/version.info" ] && chmod 444 "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/version.info" 2>/dev/null || true

    # 3. Package database file: align to 644 (owner writable, others read-only).
    [ -f "${INSTALL_ROOT}/var/ascend_package_db.info" ] && chmod 644 "${INSTALL_ROOT}/var/ascend_package_db.info" 2>/dev/null || true

    # 4. Directories: default builtin subtrees to 555, then set the run package's
    #    custom/top-level dirs to 755. The 755 set is exactly: install root, share,
    #    share/info, share/info/pto_isa, var, aarch64-linux/include/pto, and pto's
    #    direct content dirs (common, cpu, npu, npu/a2a3, npu/a5). Everything else
    #    under aarch64-linux and share/info/pto_isa/script stays 555.
    find "${INSTALL_ROOT}/aarch64-linux" -type d 2>/dev/null -exec chmod 555 {} +
    find "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/script" -type d 2>/dev/null -exec chmod 555 {} +
    # custom/top-level dirs -> 755 (matches run package CUSTOM_PERM under IS_FOR_ALL=y)
    for d in \
        "${INSTALL_ROOT}" \
        "${INSTALL_ROOT}/share" \
        "${INSTALL_ROOT}/share/info" \
        "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}" \
        "${INSTALL_ROOT}/var" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto/common" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto/cpu" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto/npu" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto/npu/a2a3" \
        "${INSTALL_ROOT}/aarch64-linux/include/pto/npu/a5"; do
        [ -d "$d" ] && chmod 755 "$d" 2>/dev/null || true
    done
fi

exit 0
