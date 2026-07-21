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
# pto-isa is a header-only library: no wheel/so to install, and no pto-specific install artifacts are
# created at install time. The top-level symlinks (bin/lib64/include/...) and the package database
# (var/ascend_package_db.info) are already handled by the cann-cmake-generated postinst body driven by
# the EngineeringCommon block in pto_isa.xml. RPM/DEB users uninstall via `rpm -e` / `dpkg -r`, so no
# extra cann_uninstall.sh entry or ascend_install.info record is needed here.
#
# Keep this script free of any literal percent sign: cann-cmake inlines it into the RPM post section,
# where RPM's spec macro processor would expand it.

exit 0
