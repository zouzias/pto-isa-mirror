# 2026-03-16 LLM Runtime Utils Migration

## Goal

Continue shrinking the hard runtime dependency from `MindSpeed` back to
`lzm_Mindspeed-LLm` after the training-entry migration.

## Scope moved into MindSpeed

The following training/runtime helpers are now owned by `MindSpeed`:

- `mindspeed/training/llm_arguments.py`
- `mindspeed/training/seed_utils.py`
- `mindspeed/tasks/utils/error_utils.py`
- `mindspeed/tasks/high_availability/high_availability_helper.py`

The active training path was rewired to use them:

- `mindspeed/training/initialize.py`
- `mindspeed/features_manager/llm_feature_adaptor.py`

## What this removed

The active `MindSpeed` training path no longer imports these modules from
`lzm_Mindspeed-LLm`:

- `mindspeed_llm.training.arguments`
- `mindspeed_llm.training.utils`
- `mindspeed_llm.tasks.utils.error_utils`
- `mindspeed_llm.tasks.high_availability.high_availability_helper`

## Remaining runtime dependency

`MindSpeed` still depends on `lzm_Mindspeed-LLm` for:

- `mindspeed.features_manager.llm_feature_registry`
  - still loads `mindspeed_llm.features_manager`
- conditional LoRA / high-availability code inside
  `mindspeed/training/llm_training.py`

Those parts are not on the critical path for the current `mc2` / `pto_mc2`
training regression, so they were left untouched in this step.

## Regression

### MC2

- run id: `migrate_more_mc2_50step`
- entry:
  - `/home/llx/MindSpeed/pretrain_jamba.py`
- steady-state window:
  - `6503-6550`
- median:
  - `2061.05 ms`
  - `39.400% MFU`

### PTO_MC2

- run id: `migrate_more_pto_mc2_50step`
- entry:
  - `/home/llx/MindSpeed/pretrain_jamba.py`
- proof of PTO load:
  - `PTO_MC2_REORDER loaded ...`
  - `PTO_MC2_REORDER first call ...`
- steady-state window:
  - `6503-6550`
- median:
  - `2092.45 ms`
  - `38.810% MFU`

## Conclusion

- The active training path depends less on `lzm_Mindspeed-LLm` than before.
- Performance stayed within the existing target band.
- `lzm_Mindspeed-LLm` still cannot be deleted yet because feature registry and
  model/task implementations still live there.
