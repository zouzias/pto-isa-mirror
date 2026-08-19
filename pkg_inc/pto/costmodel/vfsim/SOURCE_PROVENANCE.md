# VfSim native source provenance

The native simulator sources under `api/native/`, `native/`, and `configs/`
were imported from:

- Repository: https://github.com/wang-chonghao/VfSimulator
- Branch: `cpp-native-vfsim`
- Commit: `abe59b9c7cd3be8e30a41cc221f2cbc2954fed58`
- Commit date: 2026-08-19T16:40:25+08:00
- Subject: `feat: expose canonical-only native runner`

The PTO-specific lowering adapter is kept separately in `pto_adapter/` so the
upstream source layout can be refreshed without mixing PTO integration changes
into the imported native model.

## PTO integration changes

- The CMake project and option descriptions use the shortened `VfSim` name.
- `VFSIM_SOURCE_ROOT` is scoped to upstream native tests instead of being a
  public compile definition on `vfsim_native_core`.
- The native smoke-test diagnostic uses the shortened `VfSim` name.
- Upstream native regression sources are retained temporarily while native
  integration and style changes are stabilized.
- Native implementation symbols follow PTO C++ naming conventions: acronym
  types use PascalCase (`Ifu`, `Idu`, `ParamDb`, and `OooCore`), enum values and
  constants use upper snake case, and header guards end in `_H_`. Existing
  source filenames and include paths remain unchanged to keep the imported
  source layout stable.
- The canonical uarch keys are `idu_window_width`, `idu_issue_width`, and
  `ldq_width`. Readers and canonical override validation continue accepting the
  historical mixed-case spellings for compatibility.
- Native and adapter C++ sources are formatted with PTO-ISA's repository
  `.clang-format`. Imported license notices and SPDX declarations are retained.
- PTO's internal trace is lowered directly to `CanonicalVfInfo` by the
  PTO-owned `pto_canonical_lowering` component; the production adapter does
  not depend on the migration-period native `VfInfo` or
  `LegacyVfInfoAdapter`.

The native scheduling/model behavior is otherwise unchanged. PTO trace
lowering, structured status, fallback policy, logging, and relocatable config
lookup are implemented only in `pto_adapter/` and PTO CMake integration code.
