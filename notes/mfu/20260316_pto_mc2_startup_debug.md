# 2026-03-16 PTO MC2 Startup Debug

## 目标

定位 `ENABLE_PTO_MOE_MC2_REORDER=1` 训练主线在首步前后“启动不了/卡住”的具体原因，并验证 PTO-ISA 是否真实接入训练。

## 结论

当前问题被拆成了两个独立根因：

1. 首个卡点不是 PTO，也不是 MC2 dispatcher，而是 fused rotary 的 query 路径。
   - 使用默认 `--use-fused-rotary-pos-emb` 时，训练稳定卡在
     `apply_rotary_pos_emb(query)`。
   - 证据见日志：
     `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_rope_debug/train.log`
   - 关键日志：
     - `[ATTN_STAGE_DEBUG] before apply_rotary_pos_emb(query)`
     - 没有 `[ATTN_STAGE_DEBUG] after apply_rotary_pos_emb(query)`

2. 关掉 fused rotary 之后，训练可以继续往下走到 MC2/PTO 路径，但又暴露出 PTO `.so` 默认路径错误。
   - 错误路径：
     `/home/llx/pto-isa/demos/baseline/moe_grouped_ffn/build/libop_extension.so`
   - 正确路径：
     `/home/llx/pto-isa/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so`
   - 修正后，`PTO_MC2_REORDER` 明确打印成功加载。

## 关键实验

### A. fused rotary 开启

- Run:
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_rope_debug/train.log`
- 结果：
  - 所有 rank 都到达 `before model(...)`
  - `linear_qkv / q_layernorm / k_layernorm` 都返回
  - 卡在 `apply_rotary_pos_emb(query)`

### B. fused rotary 关闭，但 PTO `.so` 路径未修

- Run:
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_rope_disable_control/train.log`
- 结果：
  - `query/key rope -> core_attention -> MC2 token_permutation` 全部通过
  - 失败点变为：
    `RuntimeError: Loaded .../build/libop_extension.so, but npu::pto_mc2_reorder_chunks is missing.`

### C. fused rotary 关闭，PTO `.so` 路径修正

- Run:
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260316_pto_mc2_rope_disable_ptofix/train.log`
- 结果：
  - `ATTN_STAGE_DEBUG` 全部通过
  - `MC2_STAGE` 全部通过
  - `PTO_MC2_REORDER` 成功打印：
    - `loaded npu::pto_mc2_reorder_chunks from /home/llx/pto-isa/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so`
    - `first call input_shape=(15533,) ...`
  - 所有 rank 都打印 `after model(...)`
  - 真实训练已跑到 `6508`

## MFU

来自 run C：

- `6501`: `27397.4 ms / 2.96%`
- `6502`: `2140.5 ms / 37.94%`
- `6503`: `2090.3 ms / 38.85%`
- `6504`: `2094.5 ms / 38.77%`
- `6505`: `2098.6 ms / 38.70%`
- `6506`: `2092.8 ms / 38.80%`
- `6507`: `2093.3 ms / 38.79%`
- `6508`: `2099.8 ms / 38.67%`

中位数：

- `6501-6508`: `2096.55 ms / 38.735%`
- `6502-6508`: `2094.5 ms / 38.77%`
- `6503-6508`: `2093.9 ms / 38.78%`

## 当前判断

- “启动不了”的首个根因是 fused rotary query RoPE 路径。
- PTO-ISA `mc2_reorder` 不是首个卡点。
- 在关闭 fused rotary 并修正 `.so` 路径后，PTO-ISA 已真实接入训练主线，并可完成短窗口训练。
