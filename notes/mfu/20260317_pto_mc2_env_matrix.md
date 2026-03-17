# 2026-03-17 PTO MC2 环境变量矩阵实验记录

> 更新说明：本文中的 `defer_probs` 结论只对应“未修复实现”的早期结果。后续同日的修复与复验见 `20260317_pto_mc2_defer_probs_revival.md`。

## 目标

在当前 clean `pto_mc2` 基线之上，继续寻找还能稳定提升 MFU 的低风险方向，并保证实验过程可复盘、可归因。

## 统一实验口径

- 入口脚本：
  - `/home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- checkpoint：
  - `6500`
- 统计窗口：
  - `6503-6802`
- 固定配置：
  - `LAUNCH_PRESET=pto_mc2`
  - `USE_FUSED_ROTARY_POS_EMB=1`
- 对比基线：
  - clean `pto_mc2`
  - `2053.0 ms / 39.555% MFU`
  - 日志：`/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_clean_pto_mc2_fusedrope_long/train.log`

## 过程记录

### 阶段 1：补齐实验归因

- 修改 `run_native_baseline_resume.sh`，把以下 env 写入 `meta.txt` 和 `launch.sh`：
  - `MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE`
  - `MINDSPEED_MOE_MC2_FUSE_SHARED_EXPERTS`
  - `MINDSPEED_MOE_MC2_DISABLE_SHARED_EXPERT_OVERLAP`
- 目的：
  - 解决之前 MC2 关键开关未被 wrapper 记录的问题，避免“跑过但说不清”。

### 阶段 2：计划中的矩阵

优先按以下顺序执行：

1. `pto_mc2 + MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1`
2. `pto_mc2 + MINDSPEED_MOE_MC2_DISABLE_SHARED_EXPERT_OVERLAP=1`
3. `pto_mc2 + defer_probs + disable_shared_overlap`
4. 仅当以上仍有空间时，再考虑 `MINDSPEED_MOE_MC2_FUSE_SHARED_EXPERTS=1`

## 结果汇总

### 实验 1：`pto_mc2 + MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1`

- 命令：
  - `LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_defer_probs_long MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- 结果：
  - 训练失败，未进入可统计的稳态窗口。
- 日志：
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_long/train.log`
- 快照：
  - `meta.txt` 和 `launch.sh` 已正确记录 `MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1`
- 失败位置：
  - `mindspeed/core/transformer/moe/moe_feature/mc2moe/legacy_a2a_token_dispatcher.py`
  - `token_permutation() -> _finalize_async_metadata() -> cuda_dtoh_stream.synchronize()`
- 典型报错：
  - `RuntimeError: ... AclrtSynchronizeStreamWithTimeout(stream()), error code is 507035`
  - Ascend 日志同时给出 `vector core exception` 和 `The DDR address of the MTE instruction is out of range`
- 当前判断：
  - 这不是一个“MFU 稍差”的方向，而是当前代码/算子组合下的功能性失败。
  - 本轮不引入修复代码，避免为一个尚未证明有收益的方向堆积腐化代码。

### 实验 2：`pto_mc2 + MINDSPEED_MOE_MC2_DISABLE_SHARED_EXPERT_OVERLAP=1`

- 命令：
  - `LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_disable_shared_overlap_long MINDSPEED_MOE_MC2_DISABLE_SHARED_EXPERT_OVERLAP=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- 结果：
  - 可以稳定训练，但早期稳态已经明显落后于 clean 基线，因此在 `6521` 附近主动停止。
- 早期窗口：
  - `6503-6521`
  - `2075.0 ms / 39.14% MFU`
- 对比 clean 基线：
  - 基线：`2053.0 ms / 39.555% MFU`
  - 差值：约 `+22.0 ms / -0.415 MFU`
- 日志：
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_disable_shared_overlap_long/train.log`
- 当前判断：
  - 关闭 shared expert overlap 不是当前 `pto_mc2` 的增益方向。
  - 本轮没有为该方向引入任何代码，因此无需额外清理。

### 实验 3：`pto_mc2 + MINDSPEED_MOE_MC2_FUSE_SHARED_EXPERTS=1`

- 命令：
  - `LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_sharedfuse_long MINDSPEED_MOE_MC2_FUSE_SHARED_EXPERTS=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- 结果：
  - 可以稳定训练，但早期稳态仍低于 clean 基线，因此在 `6524` 附近主动停止。
- 早期窗口：
  - `6503-6524`
  - `2069.8 ms / 39.23% MFU`
- 对比 clean 基线：
  - 基线：`2053.0 ms / 39.555% MFU`
  - 差值：约 `+16.8 ms / -0.325 MFU`
- 日志：
  - `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_sharedfuse_long/train.log`
- 当前判断：
  - `shared_expert fuse` 在当前 `pto_mc2 + fused rotary` clean 配置下仍不是收益方向。
  - 这和历史报告结论一致，可以继续降低优先级。

## 当前轮次结论

- 本轮已经确认：
  - `defer_probs`：当前代码下功能性失败，不是可直接推进的实验面。
  - `disable_shared_expert_overlap`：负收益。
  - `fuse_shared_experts`：负收益。
- 因此，当前 `pto_mc2` clean 基线 `2053.0 ms / 39.555% MFU` 仍然是这条线的最佳已验证结果。
- 下一阶段不建议继续在现有 env 组合上穷举；更值得投入的是：
  - 做更长链路的 PTO 融合，而不是只做单个 local reorder。
  - 若要继续实验，应转向代码级优化而不是继续切 env。
