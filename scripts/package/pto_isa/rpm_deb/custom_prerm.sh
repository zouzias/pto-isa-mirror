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
# pto-isa is a header-only library: nothing to tear down beyond the install-info record.

sourcedir="${INSTALL_PATH}"
PTO_PLATFORM_DIR="pto_isa"
INFO_FILE="${sourcedir}/share/info/${PTO_PLATFORM_DIR}/ascend_install.info"

rm -f "${INFO_FILE}" 2>/dev/null || true

exit 0
