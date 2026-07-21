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
# Appended to the auto-generated prerm by cann-cmake gen_postinst_prerm.py for the rpm/deb package.
# Mirror the postinst side: remove the top-level cann_uninstall.sh entry and the install-info record
# created by custom_postinst.sh, so uninstall is clean and symmetric with the run package.

sourcedir="${INSTALL_PATH}"
PTO_PLATFORM_DIR="pto_isa"

# remove the top-level uninstall entry created by custom_postinst.sh
rm -f "${sourcedir}/cann_uninstall.sh" 2>/dev/null || true

# remove the install-info record so re-install stays clean
rm -f "${sourcedir}/share/info/${PTO_PLATFORM_DIR}/ascend_install.info" 2>/dev/null || true

exit 0
