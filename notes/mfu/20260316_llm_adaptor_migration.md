# 2026-03-16 LLM Adaptor Migration

## Scope

This change moves the LLM adaptor control path into `MindSpeed` while keeping
`lzm_Mindspeed-LLm` as a thin compatibility layer.

### Code changes

- Added `MindSpeed` registry bridge:
  - `mindspeed/features_manager/llm_feature_registry.py`
- Added `MindSpeed` adaptor:
  - `mindspeed/features_manager/llm_feature_adaptor.py`
- Reduced `lzm_Mindspeed-LLm` entry glue:
  - `mindspeed_llm/__init__.py`
  - `mindspeed_llm/tasks/megatron_adaptor_v2.py`
  - `mindspeed_llm/training/arguments.py`

## Regression

Workload:

- resume from local checkpoint iteration `6500`
- evaluate steady-state window `6503-6550`

### MC2

- log:
  - `logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/migrate_regress_mc2_50step/train.log`
- median:
  - `2062.8 ms`
  - `39.365% MFU`

### PTO_MC2

- log:
  - `logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/migrate_regress_pto_mc2_50step/train.log`
- median:
  - `2084.1 ms`
  - `38.97% MFU`
- proof of PTO load:
  - `PTO_MC2_REORDER loaded ...`
  - `PTO_MC2_REORDER first call ...`

## Conclusion

- The adaptor and feature-list control path now lives in `MindSpeed`.
- Performance stayed in the expected range after the migration.
- `lzm_Mindspeed-LLm` still cannot be deleted yet.

Remaining blockers for deleting `lzm_Mindspeed-LLm`:

- `pretrain_jamba.py`
- `mindspeed_llm.training.training`
- model/task-specific code used by the current Jamba/Qwen2 training entry
