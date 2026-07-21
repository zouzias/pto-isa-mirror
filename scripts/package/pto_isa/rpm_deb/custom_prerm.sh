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
#
# pto-isa is a header-only library: nothing pto-specific is created by custom_postinst.sh, so there is
# nothing pto-specific to tear down here. The symlinks and the package database are cleaned up by the
# cann-cmake-generated prerm body.
#
# Keep this script free of any literal percent sign: cann-cmake inlines it into the RPM preun section,
# where RPM's spec macro processor would expand it.

exit 0
