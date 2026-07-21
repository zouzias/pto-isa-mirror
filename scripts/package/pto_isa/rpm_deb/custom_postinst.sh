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
# contain any literal percent sign (no format strings, no date format, no heredocs with it). Build the
# generated uninstall entry below with plain echo redirection only.
#
# pto-isa is a header-only library: no wheel/so to install. We keep the install experience consistent
# with the makeself run package by:
#   1. touching the install-info record so the install can be tracked (run package's pto_install.sh does this)
#   2. generating the top-level cann_uninstall.sh entry so users can uninstall the same way as the run package,
#      in addition to 'rpm -e' / 'dpkg -r'.

sourcedir="${INSTALL_PATH}"
PTO_PLATFORM_DIR="pto_isa"
INFO_FILE="${sourcedir}/share/info/${PTO_PLATFORM_DIR}/ascend_install.info"

# 1. maintain install-info record (parity with run package's pto_install.sh)
if [ -d "${sourcedir}/share/info/${PTO_PLATFORM_DIR}" ]; then
    touch "${INFO_FILE}" 2>/dev/null || true
    chmod 644 "${INFO_FILE}" 2>/dev/null || true
fi

# 2. generate top-level cann_uninstall.sh entry (parity with makeself run package)
#    It delegates to share/info/pto_isa/script/pto_uninstall.sh which is shipped inside this package.
#    Written with echo redirection only, to avoid any literal percent sign in this file.
UNINSTALL_ENTRY="${sourcedir}/cann_uninstall.sh"
{
  echo '#!/bin/sh'
  echo 'SHELL_DIR=$(dirname "${BASH_SOURCE:-$0}")'
  echo 'INSTALL_PATH=$(cd "${SHELL_DIR}" && pwd)'
  echo 'TOTAL_RET=0'
  echo 'uninstall_package() {'
  echo '    path=$1'
  echo '    if [ ! -d "${INSTALL_PATH}/${path}" ]; then'
  echo '        echo "[ERROR]: ${INSTALL_PATH}/${path}: No such file or directory"'
  echo '        TOTAL_RET=1'
  echo '        return 1'
  echo '    fi'
  echo '    cd "${INSTALL_PATH}/${path}" || { TOTAL_RET=1; return 1; }'
  echo '    ./uninstall.sh'
  echo '    ret=$?'
  echo '    [ ${ret} -ne 0 ] && TOTAL_RET=1'
  echo '    return ${ret}'
  echo '}'
  echo 'if [ "$*" != "" ]; then'
  echo '    echo "[ERROR]: $*, parameter is not supported."'
  echo '    exit 1'
  echo 'fi'
  echo 'uninstall_package "share/info/pto_isa/script"'
  echo 'exit ${TOTAL_RET}'
} > "${UNINSTALL_ENTRY}"
chmod 550 "${UNINSTALL_ENTRY}" 2>/dev/null || true

exit 0
