# 2026-03-16 Training Body Migration

## Goal

Move the Jamba training entry and training host modules to `MindSpeed`, while
keeping `lzm_Mindspeed-LLm` as a compatibility layer for the remaining
task/model-specific code.

## Code movement

### New MindSpeed-owned files

- `pretrain_jamba.py`
- `mindspeed/training/__init__.py`
- `mindspeed/training/jamba_pretrain.py`
- `mindspeed/training/llm_training.py`
- `mindspeed/training/initialize.py`

### lzm compatibility wrappers

- `pretrain_jamba.py`
- `mindspeed_llm/training/training.py`
- `mindspeed_llm/training/initialize.py`

### Launcher change

- `tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
  now points `mc2` and `pto_mc2` to `MindSpeed/pretrain_jamba.py`

## Regression

Both regressions were launched from `MindSpeed/pretrain_jamba.py`.

### MC2

- run id: `migrate_trainbody_mc2_50step`
- proof:
  - `meta.txt` contains `pretrain_entry=/home/llx/MindSpeed/pretrain_jamba.py`
- steady-state window: `6503-6550`
- median:
  - `2060.55 ms`
  - `39.41% MFU`

### PTO_MC2

- run id: `migrate_trainbody_pto_mc2_50step`
- proof:
  - `meta.txt` contains `pretrain_entry=/home/llx/MindSpeed/pretrain_jamba.py`
  - training log contains:
    - `PTO_MC2_REORDER loaded ...`
    - `PTO_MC2_REORDER first call ...`
- steady-state window: `6503-6550`
- median:
  - `2087.3 ms`
  - `38.905% MFU`

## Conclusion

- The training entry and training host path now run from `MindSpeed`.
- Performance stayed in the expected range after the migration.
- `lzm_Mindspeed-LLm` still cannot be deleted yet because task/model-specific
  implementations are still imported from that repo.
