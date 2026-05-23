# dispatch_combine_moe no-AscendC 化任务

## 目标

让当前 `kernels/manual/a5/dispatch_combine_moe` 的项目源码不再直接依赖 `AscendC::` / `kernel_operator.h` 等 Ascend C 编程面。用户已确认：**不允许修改 `include/pto/**`**。

## 当前阶段

- 阶段：阶段 1-11 已完成；静态 gate、A5 compile gate、runtime gate 评估与收尾代码检视已闭环。
- 实施口径：不修改 `include/pto/**`，不做项目局部 AscendC adapter；当前项目源码最终不得直接出现 AscendC 编程面、可替代 raw builtin、host `platform_ascendc` 查询。
- 检查结论：完整静态 gate 已 PASS：AscendC、ASCENDC、kernel_operator、platform_ascendc、raw core/sync/cache builtin、icache_preload 命中均为 0；A5 compile-only gate 已 PASS。
- 最近更新时间：2026-05-24

## 已确认

- [x] 不允许扩展或修改 `include/pto/**` 公共接口。
- [x] 不能采用项目局部 adapter 藏 `AscendC::` 的方案。
- [x] compiler builtin 在 PTO 仓内有替代抓手，当前项目源码不得直接保留 raw core identity、raw event/barrier、raw cross-core sync、raw dcci/dsb 等调用。
- [x] host `platform_ascendc::PlatformAscendCManager` 在 PTO 仓内有 config/table 化替代模式，当前项目 host 侧也必须迁移。

## 待确认

- [x] 无待确认决策门；进入实施闭环。

## 待办事项

- [x] 读取 `README.md`、`design.md`、`task.md`，清理旧 A3 run mode 目标。
- [x] 读取 PTO 编程范式和 `.ai-knowledge` PTO 路由，确认纯 PTO 边界。
- [x] 盘点当前源码中的 AscendC / `kernel_operator.h` / event / core identity 触点。
- [x] 初版 no-AscendC 化设计草案写入 `design.md`。
- [x] 根据用户“不允许扩展 include/pto”反馈，更新 `design.md` 为现有 PTO-only 路线。
- [x] 根据用户“builtin 与 host 都有 PTO 仓内替代且都要替换”反馈，更新 `design.md`、`goal.md`、`task.md` 强约束口径。
- [x] 阶段 0：完成静态 no-AscendC gate 基线扫描，排除 `build/out/.cache` 等生成物；当前命中：`AscendC::` 225、`using namespace AscendC` 11、裸 `AscendC` 236、`kernel_operator.h` 11、`platform_ascendc::` 2、`PlatformAscendCManager` 4、host core query 2、raw core builtin 17、raw cache builtin 4、`icache_preload` 6。
- [x] 阶段 1：kernel entry no-`kernel_operator.h` spike，`dispatch_combine_moe.cpp` 已改为 `<pto/pto-inst.hpp>` + `AICORE` + `PTO_SYNCALL_MIX_AIC_KERNEL_META(dispatch_combine_moe, 1, 2)`；targeted 静态检查 `AscendC::`/`using namespace AscendC`/`kernel_operator.h`/AscendC tiling-task 宏命中均为 0。
- [x] 阶段 2：host resource config 替换 `platform_ascendc`，新增 `op_host/pto_resource_config.hpp`，对齐 `gemm_ar` 的 `CONFIG_*` / table 化资源描述方式；targeted 静态检查 `platform_ascendc`/`PlatformAscendCManager`/host core query/include 命中均为 0。
- [x] 阶段 3：迁移 core identity / AIC-AIV dispatch 到 `pto::SYNCALL_GET_MIX_*`；新增 `PtoMix*` / `PtoAic*` / `PtoAiv*` helper，raw `get_block_idx/get_block_num/get_subblockid/get_subblockdim/block_idx` 静态检查命中 0。
- [x] 阶段 4：迁移 `utils/moe_pto_utils.hpp`，集中改为 `pto::Event`、`pto::TSYNC`、`pto::SYNCALL<pto::SyncCoreType::Mix>`、`pto::TSync_Custom` 等现有 PTO sync 抓手。
- [x] 阶段 5：迁移 MMAD/vector/epilogue helper 中的 AscendC sync/barrier，不保留 raw `pipe_barrier/set_flag/wait_flag`。
- [x] 阶段 6：迁移 routing 和 unpermute 子系统，去除 `AscendC::TPipe`、`using namespace AscendC`、raw core idx 和 HardEvent 依赖。
- [x] 阶段 7：迁移 remote window cache/sync，去除 `AscendC::GlobalTensor/DataCacheCleanAndInvalid` 与 raw `dcci/dsb`，使用 PTO comm signal / PTO soft visibility helper。
- [x] 阶段 8：静态 gate：源码范围无设计禁止项；输出 `STATIC_GATE_RESULT=PASS`。
- [x] 阶段 9：compile gate：A5 compile-only 已通过；命令为加载 `/home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh` 后执行 `cmake --build /tmp/dispatch_combine_moe_a5_compile_gate --target dispatch_combine_moe -j8`，输出 `[100%] Built target dispatch_combine_moe`。同时复扫非法 `TSYNC<pto::Op::VECTOR/TMATMUL/...>` 命中 0，`include/pto/**` git diff 为空。
- [x] 阶段 10：黑盒 runtime gate：已执行环境评估，`npu-smi info` 显示当前设备为 910B1/A3；本任务为 A5 项目且当前口径是 A5 compile-only，因此不声明也不执行 A5 runtime。未发现 NPU 进程占用，但不在 A3 上强跑 A5 用例。
- [x] 阶段 11：收尾代码检视：已检查 PTO 替代真实性、AscendC/ASCENDC 泄漏、算法/overlap 降级、sync 卡死风险、是否误改 `include/pto/**`。独立 verifier 给出静态 gate PASS、`include/pto/**` diff 为空、A5 compile PASS；用户指出大写 `ASCENDC` 残留后，已将 include guard 改为 `PTO_*`，`ASCENDC_ASSERT` 改为 PTO `PTO_ASSERT`，`ASCENDC_OOM` 改为项目本地 `PTO_DISPATCH_COMBINE_MOE_OOM_CHECK`，复扫 `ASCENDC_MATCHES=0`，A5 compile PASS。`DESIGN.md` 清空按用户明确“不需要管”处理。残留风险：sync 等价和性能/overlap 仍只有静态+编译证据，A5 runtime 未在当前 A3 机器验证。
- [x] 追加代码检视发现项已修复：`CombineV2` 改用 `PtoAivPairedAicIdx()` 保持原 AIV `get_block_idx()` 语义；`PtoGetSortLen/PtoGetSortOffset` 改为 `PTO_SORT_RECORD_BYTES / sizeof(T)` 计算 packed 8-byte record 的元素数；`PtoSetWaitFlag<Event>(PtoHardEvent)` 改为显式 `PtoDefaultEventId()` 映射，避免 enum 值直接作为 event id；host resource config 对空默认、`Ascend950`、`Ascend950DT_*`、`Ascend950PR_*` 做显式支持，其他 soc 报错且仍支持 `CONFIG_DISPATCH_COMBINE_MOE_*` 宏覆盖。复验：A5 compile PASS、no-AscendC/ASCENDC static gate PASS、`include/pto/**` diff 为空，独立 verifier 推荐 pass。

## 边界

- 不修改 `README.md` 原始 spec，除非用户后续明确要求同步文档。
- 不修改 `include/pto/**`。
- 不恢复 naive 单 AIV correctness path。
- 不为了白盒测试新增测试代码；中间态以代码检视 + 静态/编译 gate 为主。
- 不把 AscendC 继续藏在当前项目局部 adapter 中。
