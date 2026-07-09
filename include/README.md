# include/

Public C/C++ headers for PTO ISA (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See [include/pto/README.md](pto/README.md) for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu / comm)

## Related Docs

- [ISA guide](../docs/README.md)
- [Getting started](../docs/getting-started.md)

## PTO Instruction Implementation Status (CPU / Costmodel / A2 / A3 / A5 / Kirin)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend). More information about this backend can be found in [docs/coding/cpu_sim.md](../docs/coding/cpu_sim.md)
- **Costmodel**: `__COSTMODEL` (A2 / A3 cost model backend, including `stub` and `fit` paths; if either path supports an instruction, it is marked as supported).
- **A2 (Ascend 910B) / A3 (Ascend 910C)**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **A5 (Ascend 950)**: uses the `include/pto/npu/a5/` implementation.
- **Kirin**: uses the `include/pto/npu/kirin9030/` implementation.

| Instruction | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER.md) | Yes | No | Yes | Yes | Yes | No |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | Yes | No | Yes | Yes | Yes | No |
| [`SETFMATRIX`](../docs/isa/SETFMATRIX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`SET_IMG2COL_PADDING`](../docs/isa/SET_IMG2COL_PADDING.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`SET_IMG2COL_RPT`](../docs/isa/SET_IMG2COL_RPT.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`SET_QUANT_SCALAR`](../docs/isa/SET_QUANT_SCALAR.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`SET_QUANT_VECTOR`](../docs/isa/SET_QUANT_VECTOR.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`SYNCALL`](../docs/isa/SYNCALL.md) | No | No | Yes | Yes | Yes | No |
| [`TABS`](../docs/isa/TABS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADD`](../docs/isa/TADD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADDC`](../docs/isa/TADDC.md) | Yes | No | No | No | No | No |
| [`TADDDEQRELU`](../docs/isa/TADDDEQRELU.md) | No | No | Yes | Yes | Yes | No |
| [`TADDReluConv`](../docs/isa/TADDReluConv.md) | No | No | Yes | Yes | Yes | No |
| [`TADDS`](../docs/isa/TADDS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TADDSC`](../docs/isa/TADDSC.md) | Yes | No | No | No | No | No |
| [`TALLOC`](../docs/isa/TALLOC.md) | No | Yes | Yes | Yes | Yes | No |
| [`TAND`](../docs/isa/TAND.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TANDS`](../docs/isa/TANDS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TASSIGN`](../docs/isa/TASSIGN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TAXPY`](../docs/isa/TAXPY.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TBROADCAST`](../docs/isa/comm/TBROADCAST.md) | Yes | No | Yes | Yes | Yes | No |
| [`TCI`](../docs/isa/TCI.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCMP`](../docs/isa/TCMP.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCMPS`](../docs/isa/TCMPS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLARGMAX`](../docs/isa/TCOLARGMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLARGMIN`](../docs/isa/TCOLARGMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDADD`](../docs/isa/TCOLEXPANDADD.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDDIV`](../docs/isa/TCOLEXPANDDIV.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDEXPDIF`](../docs/isa/TCOLEXPANDEXPDIF.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDMAX`](../docs/isa/TCOLEXPANDMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDMIN`](../docs/isa/TCOLEXPANDMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDMUL`](../docs/isa/TCOLEXPANDMUL.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLEXPANDSUB`](../docs/isa/TCOLEXPANDSUB.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLPROD`](../docs/isa/TCOLPROD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TCONCAT`](../docs/isa/TCONCAT.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TCVT`](../docs/isa/TCVT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TDEINTERLEAVE`](../docs/isa/TDEINTERLEAVE.md) | No | No | No | No | Yes | No |
| [`TDIV`](../docs/isa/TDIV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TDIVS`](../docs/isa/TDIVS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXP`](../docs/isa/TEXP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TEXTRACT_FP`](../docs/isa/TEXTRACT_FP.md) | No | No | Yes | Yes | Yes | Yes |
| [`TFILLPAD`](../docs/isa/TFILLPAD.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TFILLPAD_EXPAND`](../docs/isa/TFILLPAD_EXPAND.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TFILLPAD_INPLACE`](../docs/isa/TFILLPAD_INPLACE.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TFMOD`](../docs/isa/TFMOD.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TFMODS`](../docs/isa/TFMODS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TFREE`](../docs/isa/TFREE.md) | No | Yes | Yes | Yes | Yes | No |
| [`TFUSEDMULADD`](../docs/isa/TFUSEDMULADD.md) | No | No | Yes | Yes | Yes | No |
| [`TFUSEDMULADDRELU`](../docs/isa/TFUSEDMULADDRELU.md) | No | No | Yes | Yes | Yes | No |
| [`TGATHER`](../docs/isa/TGATHER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TGEMV`](../docs/isa/TGEMV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGEMV_ACC`](../docs/isa/TGEMV_ACC.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGEMV_BIAS`](../docs/isa/TGEMV_BIAS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TGEMV_MX`](../docs/isa/TGEMV_MX.md) | Yes | No | No | No | Yes | No |
| [`TGET`](../docs/isa/comm/TGET.md) | Yes | No | Yes | Yes | Yes | No |
| [`TGET_ASYNC`](../docs/isa/comm/TGET_ASYNC.md) | Yes | No | Yes | Yes | Yes | No |
| [`TGET_SCALE_ADDR`](../docs/isa/TGET_SCALE_ADDR.md) | Yes | Yes | No | No | Yes | Yes |
| [`TIMG2COL`](../docs/isa/TIMG2COL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TINSERT`](../docs/isa/TINSERT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TINSERT_FP`](../docs/isa/TINSERT_FP.md) | No | No | Yes | Yes | Yes | Yes |
| [`TINTERLEAVE`](../docs/isa/TINTERLEAVE.md) | No | No | No | No | Yes | No |
| [`TLOAD`](../docs/isa/TLOAD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TLOG`](../docs/isa/TLOG.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TLRELU`](../docs/isa/TLRELU.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMATMUL_MX`](../docs/isa/TMATMUL_MX.md) | Yes | No | No | No | Yes | Yes |
| [`TMAX`](../docs/isa/TMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMAXS`](../docs/isa/TMAXS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMIN`](../docs/isa/TMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMINS`](../docs/isa/TMINS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV`](../docs/isa/TMOV.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMUL`](../docs/isa/TMUL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TMULADDDST`](../docs/isa/TMULADDDST.md) | No | No | Yes | Yes | Yes | No |
| [`TMULS`](../docs/isa/TMULS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNEG`](../docs/isa/TNEG.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNOT`](../docs/isa/TNOT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TNOTIFY`](../docs/isa/comm/TNOTIFY.md) | Yes | No | Yes | Yes | Yes | No |
| [`TOR`](../docs/isa/TOR.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TORS`](../docs/isa/TORS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPairReduceSum`](../docs/isa/TPairReduceSum.md) | No | No | Yes | Yes | Yes | No |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPARTARGMAX`](../docs/isa/TPARTARGMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPARTARGMIN`](../docs/isa/TPARTARGMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPARTMUL`](../docs/isa/TPARTMUL.md) | No | No | Yes | Yes | Yes | Yes |
| [`TPOP`](../docs/isa/TPOP.md) | Yes | Yes | Yes | Yes | Yes | No |
| [`TPOW`](../docs/isa/TPOW.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPOWS`](../docs/isa/TPOWS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPREFETCH`](../docs/isa/TPREFETCH.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TPREFETCH_ASYNC`](../docs/isa/TPREFETCH_ASYNC.md) | Yes | No | Yes | Yes | Yes | No |
| [`TPRELU`](../docs/isa/TPRELU.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TPRINT`](../docs/isa/TPRINT.md) | Yes | Yes | Yes | Yes | Yes | No |
| [`TPUSH`](../docs/isa/TPUSH.md) | Yes | Yes | Yes | Yes | Yes | No |
| [`TPUT`](../docs/isa/comm/TPUT.md) | Yes | No | Yes | Yes | Yes | No |
| [`TPUT_ASYNC`](../docs/isa/comm/TPUT_ASYNC.md) | Yes | No | Yes | Yes | Yes | No |
| [`TQUANT`](../docs/isa/TQUANT.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TRANDOM`](../docs/isa/TRANDOM.md) | Yes | No | No | No | Yes | No |
| [`TRECIP`](../docs/isa/TRECIP.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TREDUCE`](../docs/isa/comm/TREDUCE.md) | Yes | No | Yes | Yes | Yes | No |
| [`TRELU`](../docs/isa/TRELU.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TREM`](../docs/isa/TREM.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TREMS`](../docs/isa/TREMS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWARGMAX`](../docs/isa/TROWARGMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWARGMIN`](../docs/isa/TROWARGMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWEXPANDADD`](../docs/isa/TROWEXPANDADD.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDEXPDIF`](../docs/isa/TROWEXPANDEXPDIF.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDMAX`](../docs/isa/TROWEXPANDMAX.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDMIN`](../docs/isa/TROWEXPANDMIN.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWPROD`](../docs/isa/TROWPROD.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSEL`](../docs/isa/TSEL.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSELS`](../docs/isa/TSELS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSHL`](../docs/isa/TSHL.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSHLS`](../docs/isa/TSHLS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSHR`](../docs/isa/TSHR.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSHRS`](../docs/isa/TSHRS.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSORT32`](../docs/isa/TSORT32.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSQRT`](../docs/isa/TSQRT.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSTORE`](../docs/isa/TSTORE.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TSUB`](../docs/isa/TSUB.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUBC`](../docs/isa/TSUBC.md) | Yes | No | No | No | No | No |
| [`TSUBRELU`](../docs/isa/TSUBRELU.md) | No | No | Yes | Yes | Yes | No |
| [`TSUBS`](../docs/isa/TSUBS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | Yes | No | No | No | No | No |
| [`TSUBVIEW`](../docs/isa/TSUBVIEW.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TSYNC`](../docs/isa/TSYNC.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TTEST`](../docs/isa/comm/TTEST.md) | Yes | No | Yes | Yes | Yes | No |
| [`TTRANS`](../docs/isa/TTRANS.md) | Yes | Yes | Yes | Yes | Yes | Yes |
| [`TTRI`](../docs/isa/TTRI.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TWAIT`](../docs/isa/comm/TWAIT.md) | Yes | No | Yes | Yes | Yes | No |
| [`TXOR`](../docs/isa/TXOR.md) | Yes | No | Yes | Yes | Yes | Yes |
| [`TXORS`](../docs/isa/TXORS.md) | Yes | No | Yes | Yes | Yes | Yes |

Notes:

- `Yes`: the backend has an available implementation.
- `TODO`: the instruction is already part of the public API or documented ISA surface, but the backend implementation is not available yet or has not been integrated for that backend.
- `No`: explicitly unsupported, or not planned for that backend at the moment.
- Blank: the status has not been finalized yet, or the entry is still being reviewed.
