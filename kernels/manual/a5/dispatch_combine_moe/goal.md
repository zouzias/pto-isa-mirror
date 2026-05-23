# dispatch_combine_moe no-AscendC 化目标

## 2026-05-23 初始目标

让 `kernels/manual/a5/dispatch_combine_moe` 的当前项目源码不再直接依赖 `AscendC::`、`using namespace AscendC`、`kernel_operator.h` 等 Ascend C 编程面。

约束：

- 不允许修改 `include/pto/**`。
- 不采用当前项目局部 adapter 藏 `AscendC::` 的方案。
- 当前项目源码中的大写 `ASCENDC` 关键字也必须清零，包括 include guard、断言宏和 OOM 宏名残留。
- 当前项目源码中可替代的底层 builtin 也必须替换为 PTO 仓已有接口或已有模式，包括 core identity、event/barrier、cross-core sync、cache/visibility、comm signal 等。
- host 侧 `platform_ascendc::PlatformAscendCManager`、`GetCoreNumAiv/GetCoreNumAic/CalcTschBlockDim` 也必须迁移到 PTO 仓内已有的 config/table 化资源描述方式。
- 优先保持 mixed AIC/AIV MegaMoE 主线、PTO tile/vector/comm pipeline、HCCL remote window 数据流，不回退到 naive 单 AIV correctness path。
- 中间态以设计检视、静态 gate、编译 gate 和代码检视闭环；A5 runtime 不声明，A3 runtime 按可用性执行。

## 2026-05-23 口径刷新

用户已确认：compiler builtin 与 host `platform_ascendc` 在 PTO 仓内都有可替代抓手，本任务不保留直接调用。实施时以 `include/pto/**` 现有能力和 `gemm_ar` 等现有 PTO 算子模式为依据，不扩展 PTO 公共层。
