#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${SCRIPT_DIR}/../build"

set -euo pipefail
python3 "${SCRIPT_DIR}/../scripts/pipeline_log_analysis_detailed.py" \
	--device-addrs "${build}/device_addrs.toml" \
	--cube-start "${build}/core0.cubecore0.instr_popped_log.dump" \
	--cube-end "${build}/core0.cubecore0.instr_log.dump" \
	--vec-start "${build}/core0.veccore0.instr_popped_log.dump" \
	--vec-end "${build}/core0.veccore0.instr_log.dump" \
	--cube-mte1 "${build}/core0.cubecore0.ccu.mte1_issque.dump" \
	--cube-mte2 "${build}/core0.cubecore0.ccu.mte2_issque.dump" \
	--cube-mte3 "${build}/core0.cubecore0.ccu.mte3_issque.dump" \
	--cube-fixp "${build}/core0.cubecore0.ccu.fixp_issque.dump" \
	--cube-cube "${build}/core0.cubecore0.ccu.cube_issque.dump" \
	--vec-mte2 "${build}/core0.veccore0.ccu.mte2_issque.dump" \
	--vec-mte3 "${build}/core0.veccore0.ccu.mte3_issque.dump" \
	--vec-vec "${build}/core0.veccore0.ccu.vec_issque.dump" \
	--out-csv timeline_detailed.csv \
	--out-json timeline_detailed.json \
	--out-agg timeline_detailed_agg.csv \
	--out-svg timeline_detailed.svg
