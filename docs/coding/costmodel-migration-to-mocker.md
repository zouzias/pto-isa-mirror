# `__COSTMODEL` Trace Backend Migration

## Goal

Remove the legacy predictor backend and make `__COSTMODEL` use the host-side trace backend.

After this migration, `__COSTMODEL` is the only host-side trace switch. The trace backend now lives under
`include/pto/costmodel/`.

## File-By-File Checklist

### Core wiring

- `include/pto/pto-inst.hpp`
  Route `__COSTMODEL` to `pto/costmodel/runtime_stub.hpp` instead of `pto/common/cpu_stub.hpp`.

- `include/pto/common/pto_instr.hpp`
  Make `__COSTMODEL` own the trace wrapping path directly.

- `include/pto/common/pto_instr_impl.hpp`
  Remove the old `#ifdef __COSTMODEL` include block for `include/pto/costmodel/a2a3/*.hpp`.

### Remove old costmodel-only behavior

- `include/pto/common/pto_tile.hpp`
  Remove `cycle`, `SetCycle()`, `GetCycle()`, and remove the `__COSTMODEL`-only `TileLeft` alias behavior.

- `include/pto/common/constants.hpp`
  Remove the `__COSTMODEL` special case so host-runtime BF16 behavior follows normal costmodel trace rules.

- `include/pto/comm/pto_comm_instr_impl.hpp`
  Remove the `__COSTMODEL` special case so comm selection no longer treats it differently.

### Delete obsolete backend and runners

- Delete the legacy predictor sources and replace them with the renamed trace backend folder.
- Delete `tests/run_costmodel.py`
- Delete `tests/run_costmodel_tests.sh`

### Build, package, and docs

- `build.sh`
  Remove the old costmodel test invocation.

- `scripts/package/pto_isa/pto_isa.xml`
  Package the renamed `include/pto/costmodel` trace backend folder.

- `include/README.md`
  Redefine `__COSTMODEL` as the public trace backend switch.

- `include/README_zh.md`
  Same update in Chinese.

- Delete `docs/coding/costmodel-roadmap.md`
  It documents the backend being removed.

### Test helper cleanup

- `tests/common/test_common.h`
  Remove legacy `__COSTMODEL` branches so test helpers no longer preserve old backend behavior.

## Expected End State

- `__COSTMODEL` compiles through the renamed costmodel runtime and trace/evaluator path.
- The public trace backend folder is `include/pto/costmodel/`.
- Dedicated costmodel trace tests live under `tests/costmodel/`.
- No legacy costmodel-only tile/runtime semantics remain.
