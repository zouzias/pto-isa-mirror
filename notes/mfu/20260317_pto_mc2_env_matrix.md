# 2026-03-17 PTO MC2 环境变量矩阵实验记录

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

待补充。
