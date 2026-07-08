# CCU DSL TGather E2E Validation

This page records the end-to-end validation flow for the experimental A5
`comm/tgather_ccu_dsl` ST. The testcase compiles `tgather_ccu.dsl.cpp` into
microcode with a `sotac` executable, then uses the CCU offline adapter C ABI to
register and launch that microcode through the hcomm runtime.

Install hcomm-9.1.0 into the CANN/HCOMM environment before running the test:

```bash
git clone https://gitcode.com/cann/hcomm.git hcomm-9.1.0
cd hcomm-9.1.0
# Follow the hcomm-9.1.0 repository instructions to build and install it.
```

Download the SotaCompiler `sotac` executable from `sotac_link` and pass its path
through `PTO_SOTAC_EXECUTABLE`. For current PR validation, the script below
builds and uses the adapter `fake_sotac` helper, because the real `sotac`
executable is not available yet.

The adapter is consumed through `ccu_offline_adapter_c.h` and
`libccu_offline_adapter_hcomm.so`. The adapter may keep its internal
implementation in C++. The adapter repository used by this validation flow is
`https://gitcode.com/Kevin673/adapter.git`.

The script uses `CANN_ROOT` for CANN/HCOMM paths. It does not set or overwrite
other Ascend environment variables required by the local test environment.

Required environment:

| Variable | Description |
| --- | --- |
| `CANN_ROOT` | CANN root containing the hcomm-9.1.0 install, for example `/path/to/Ascend_624/cann`. |
| `ADAPTER_ROOT` | Adapter source tree from `https://gitcode.com/Kevin673/adapter.git`. Defaults to `../adapter` relative to the PTO repo. |
| `LIBHCOMM_PATH` | hcomm headers/private shim root. Defaults to `${CANN_ROOT}/x86_64-linux`. |
| `GTEST_ROOT` | Old-ABI GoogleTest prefix used by adapter real-runtime tests. Defaults to `${HOME}/gtest`. |
| `CCU_GTEST_ROOT` | Old-ABI GoogleTest prefix used by PTO CCU ST. Defaults to `${GTEST_ROOT}`. |
| `MPI_LIB_PATH` | MPI shared library used by adapter `run.sh`. Defaults to `/lib/x86_64-linux-gnu/libmpich.so`. |

Example for creating and running the script:

```bash
cd /path/to/pto-isa_main
mkdir -p /tmp/pto_ccu_dsl_e2e
awk '/^<!-- BEGIN_E2E_SCRIPT -->/{flag=1;next}/^<!-- END_E2E_SCRIPT -->/{flag=0}flag' \
  docs/isa/comm/CCU_DSL_E2E.md | sed '1d;$d' > /tmp/pto_ccu_dsl_e2e/e2e_pr.sh
CANN_ROOT=/path/to/Ascend_624/cann bash /tmp/pto_ccu_dsl_e2e/e2e_pr.sh
```

After building the adapter artifacts, the PTO ST can also be launched directly
through the communication test wrapper:

```bash
bash tests/run_comm_test.sh -v a5 -n 2 -t tgather_ccu_dsl
```

Script:

<!-- BEGIN_E2E_SCRIPT -->
```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
  PTO_ISA_ROOT="$(cd "${PTO_ISA_ROOT}" && pwd)"
else
  PTO_ISA_ROOT="$(pwd)"
fi
ADAPTER_ROOT="${ADAPTER_ROOT:-$(cd "${PTO_ISA_ROOT}/../adapter" && pwd)}"
JOBS="${JOBS:-$(nproc)}"

require_dir() {
  local name="$1"
  local path="$2"
  if [[ -z "${path}" || ! -d "${path}" ]]; then
    echo "error: ${name} is not a directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local name="$1"
  local path="$2"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "error: ${name} is not a file: ${path}" >&2
    exit 1
  fi
}

if [[ -z "${CANN_ROOT:-}" ]]; then
  echo "error: CANN_ROOT is required, for example /path/to/Ascend_624/cann" >&2
  exit 1
fi

require_dir PTO_ISA_ROOT "${PTO_ISA_ROOT}"
require_dir ADAPTER_ROOT "${ADAPTER_ROOT}"
require_dir CANN_ROOT "${CANN_ROOT}"

LIBHCOMM_PATH="${LIBHCOMM_PATH:-${CANN_ROOT}/x86_64-linux}"
GTEST_ROOT="${GTEST_ROOT:-${HOME}/gtest}"
CCU_GTEST_ROOT="${CCU_GTEST_ROOT:-${GTEST_ROOT}}"
MPI_LIB_PATH="${MPI_LIB_PATH:-/lib/x86_64-linux-gnu/libmpich.so}"

require_dir LIBHCOMM_PATH "${LIBHCOMM_PATH}"
require_dir GTEST_ROOT "${GTEST_ROOT}"
require_file MPI_LIB_PATH "${MPI_LIB_PATH}"

HCOMM_OFFLINE_BUILD_DIR="${HCOMM_OFFLINE_BUILD_DIR:-${ADAPTER_ROOT}/build}"
if [[ "${HCOMM_OFFLINE_BUILD_DIR}" != /* ]]; then
  HCOMM_OFFLINE_BUILD_DIR="${ADAPTER_ROOT}/${HCOMM_OFFLINE_BUILD_DIR}"
fi

export LIBHCOMM_PATH
export GTEST_ROOT
export CCU_GTEST_ROOT
export MPI_LIB_PATH
export HCOMM_OFFLINE_BUILD_DIR
export HCOMM_OFFLINE_RANK0_MICROCODE="${HCOMM_OFFLINE_RANK0_MICROCODE:-${ADAPTER_ROOT}/fs0.txt}"
export HCOMM_OFFLINE_RANK1_MICROCODE="${HCOMM_OFFLINE_RANK1_MICROCODE:-${ADAPTER_ROOT}/fs1.txt}"
export HCOMM_OFFLINE_DUMP_MICROCODE="${HCOMM_OFFLINE_DUMP_MICROCODE:-1}"
export HCOMM_OFFLINE_DUMP_MOCK_PATCH="${HCOMM_OFFLINE_DUMP_MOCK_PATCH:-0}"
export HCOMM_OFFLINE_AG_HBM_PROBE="${HCOMM_OFFLINE_AG_HBM_PROBE:-1}"
export HCOMM_OFFLINE_AG_HBM_VALIDATE="${HCOMM_OFFLINE_AG_HBM_VALIDATE:-1}"
export HCOMM_OFFLINE_AG_HBM_RANDOM_SEED="${HCOMM_OFFLINE_AG_HBM_RANDOM_SEED:-0x9e3779b97f4a7c15}"
export HCOMM_OFFLINE_ENABLE_NATIVE_MPI_PIPELINE_TEST="${HCOMM_OFFLINE_ENABLE_NATIVE_MPI_PIPELINE_TEST:-1}"

export PTO_SOTAC_EXECUTABLE="${PTO_SOTAC_EXECUTABLE:-${HCOMM_OFFLINE_BUILD_DIR}/fake_sotac}"
export PTO_CCU_ADAPTER_SOURCE_DIR="${PTO_CCU_ADAPTER_SOURCE_DIR:-${ADAPTER_ROOT}}"
export PTO_CCU_ADAPTER_SO="${PTO_CCU_ADAPTER_SO:-${HCOMM_OFFLINE_BUILD_DIR}/hcomm/libccu_offline_adapter_hcomm.so}"
export PTO_CCU_ADAPTER_INCLUDE_DIR="${PTO_CCU_ADAPTER_INCLUDE_DIR:-${ADAPTER_ROOT}/hcomm/include}"

if [[ -f "${CANN_ROOT}/set_env.sh" ]]; then
  source "${CANN_ROOT}/set_env.sh"
else
  echo "warning: ${CANN_ROOT}/set_env.sh not found; using current environment" >&2
fi

export LD_LIBRARY_PATH="${HCOMM_OFFLINE_BUILD_DIR}/hcomm:${CANN_ROOT}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

cmake_cxx_flags="\
 -I${CANN_ROOT}/x86_64-linux/pkg_inc/runtime\
 -I${CANN_ROOT}/x86_64-linux/pkg_inc\
 -I${LIBHCOMM_PATH}/include/hcomm\
 -I${CANN_ROOT}/x86_64-linux/asc/impl/adv_api/detail/hccl/cc/src/aicpu_kfc/pub_inc\
 -I${CANN_ROOT}/x86_64-linux/pkg_inc/base\
 -I${CANN_ROOT}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc\
 -I${LIBHCOMM_PATH}"

cmake -S "${ADAPTER_ROOT}" -B "${HCOMM_OFFLINE_BUILD_DIR}" \
  -DHCOMM_PATH="${CANN_ROOT}" \
  -DLIBHCOMM_PATH="${LIBHCOMM_PATH}" \
  -DHCOMM_OFFLINE_USE_HCOMM_RUNTIME=ON \
  -DHCOMM_OFFLINE_ADAPTER_BUILD_SHARED=ON \
  -DHCOMM_OFFLINE_BUILD_FAKE_SOTAC=ON \
  -DHCOMM_OFFLINE_GTEST_ROOT="${GTEST_ROOT}" \
  -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}"

cmake --build "${HCOMM_OFFLINE_BUILD_DIR}" --target fake_sotac -j"${JOBS}"
cmake --build "${HCOMM_OFFLINE_BUILD_DIR}" --target ccu_offline_adapter_hcomm_shared -j"${JOBS}"

require_file PTO_SOTAC_EXECUTABLE "${PTO_SOTAC_EXECUTABLE}"
require_file PTO_CCU_ADAPTER_SO "${PTO_CCU_ADAPTER_SO}"
require_file ccu_offline_adapter_c.h "${PTO_CCU_ADAPTER_INCLUDE_DIR}/ccu_offline_adapter_c.h"

file "${PTO_SOTAC_EXECUTABLE}"
ls -l "${PTO_CCU_ADAPTER_SO}"

bash "${ADAPTER_ROOT}/run.sh"

cd "${PTO_ISA_ROOT}"
bash tests/run_comm_test.sh -v a5 -n 2 -t tgather_ccu_dsl
```
<!-- END_E2E_SCRIPT -->
