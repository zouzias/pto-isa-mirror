---
name: card-table-pto-primitives-map
description: |
  触发：查找 PTO Primitives Map 对应的映射表、规格表或入口对照时。
  Trigger: Lookup the PTO Primitives Map mapping table, spec table, or entry cross-reference.
card_type: table
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO Primitives Map

## purpose
- 把原语族、源码入口、文档入口、kernel 模式与 testcase 入口收成一张最小映射表。
- 不重复 `docs/isa` 的单指令说明，只负责知识路由。
- 读 PTO 原语时同时看三件事：`manifest category`、`include/README` 支持矩阵、代表性 kernel/test。

| Primitive family | Manifest category cue | Impl-status cue | Primary source | Representative docs | Representative kernels | Representative tests | Notes |
|---|---|---|---|---|---|---|---|
| Unified entry | route | `include/README.md` entry | `include/pto/pto-inst.hpp` | `docs/PTOISA.md` | - | - | PTO include 第一跳 |
| Common API | route | shared across backends | `include/pto/common/pto_instr.hpp` | `docs/isa/manifest.yaml` | - | - | 权威公共 API 面 |
| Impl dispatch | route | backend split | `include/pto/common/pto_instr_impl.hpp` | - | - | - | 分派到 npu/cpu/costmodel/comm |
| Load / Store | data movement | `TLOAD/TSTORE` A2/A3/A5 = Yes | `include/pto/common/pto_instr.hpp` | `docs/isa/*TLOAD*`, `*TSTORE*` | `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp` | `tests/npu/a2a3/src/st/testcase/tload_gm2mat/*`, `tests/npu/a2a3/src/st/testcase/tstore_mat2gm/*`, `tests/npu/a5/src/st/testcase/tload_mx_NZ/*` | 最容易和 layout/Tile 绑定 |
| Manual binding / sync | Manual / Resource Binding | `TASSIGN` A2/A3/A5 = Yes, `TSYNC` CPU = TODO | `include/pto/common/type.hpp`, `event.hpp`, `fifo.hpp` | `docs/isa/*TASSIGN*`, `*TSYNC*` | `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp` | `tests/npu/a5/src/st/testcase/tadd/*`, `tests/npu/a2a3/src/st/testcase/tadd/*` | `TASSIGN`、事件、flag/wait |
| Vec elementwise / reduce | Elementwise / Reduce / Gather | `TADD/TROW*/TGATHER` broadly available | `include/pto/common/pto_instr.hpp` | `docs/isa/*TADD*`, `*TROW*`, `*TCOL*`, `*TGATHER*` | `kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp`, `kernels/manual/a2a3/topk/topk_kernel.cpp` | `tests/npu/a5/src/st/testcase/tadd/*`, `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp` | TopK / Softmax 高频命中 |
| Matmul family | Matmul / Extract / Img2Col | `TMATMUL*` / `TEXTRACT` broadly available | `include/pto/common/pto_instr.hpp` | `docs/isa/*TMATMUL*`, `*TEXTRACT*`, `*TIMG2COL*` | `kernels/manual/common/flash_atten/pto_macro_matmul.hpp`, `kernels/manual/a5/matmul_mxfp4_performance/mxmatmul_performance_kernel.cpp` | `tests/npu/a5/src/st/testcase/tmatmul/*`, `tests/npu/a5/src/st/testcase/ttrans_conv/*` | cube 路线主入口 |
| Comm primitives | communication | see `include/README.md` + `include/pto/comm` | `include/pto/comm/pto_comm_inst.hpp` | `docs/isa/comm/*` | `kernels/manual/a5/allgather_gemm/*`, `kernels/manual/a2a3/gemm_ar/*` | `tests/npu/a5/comm/st/testcase/tgather/tgather_kernel.cpp` | `TPUT/TGET/TGATHER/...` |
| Quant path | quant / dequant / format transform | A5 coverage richer than A2/A3 | `include/pto/common/pto_instr.hpp` + backend | `docs/isa/*TQUANT*`, `*TDEQUANT*` | `kernels/manual/a5/matmul_mxfp4_performance/*` | `tests/npu/a5/src/st/testcase/tquant/*`, `tests/npu/a2a3/src/st/testcase/tquant/*` | A5 路径更丰富 |

## related
- `wiki/topics/pto-overview.md`
- `wiki/concepts/pto-layering.md`
- `wiki/topics/pto-isa-mapping.md`
- `cards/tables/pto-kernel-pattern-map.md`
- `cards/tables/pto-test-entry-map.md`
