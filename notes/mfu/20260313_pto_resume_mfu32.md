# PTO-ISA MoE Grouped FFN Resume MFU (2026-03-13)

This note records the observed training MFU when forcing the PTO-ISA MoE grouped-FFN stage1 hook.

## Setup

- Model/run: `qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0`
- Method: resume from checkpoint at iter 6500 and run 20 iterations (6500 -> 6520) to reduce cold-start noise.
- MFU window: iter 6503-6520 (steady state)

## Results

- Baseline (`ENABLE_PTO_MOE_GROUPED_FFN=0`):
  - MFU median: 37.24%
  - Iter time median: 2180.7 ms
- PTO strict (`ENABLE_PTO_MOE_GROUPED_FFN=1`, `PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT=1`):
  - MFU median: 31.88% (about 32%)
  - Iter time median: 2547.0 ms

## Logs

- Baseline: `/home/llx/pto-isa/logs_pto_resume_test/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260313_190729/baseline_noparams3.log`
- PTO: `/home/llx/pto-isa/logs_pto_resume_test/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260313_190729/pto_noparams.log`

## Notes

- Enabling `--log-params-norm` triggered an `HcclAllreduce` stream allocation failure in this environment; it was disabled for the numbers above.

