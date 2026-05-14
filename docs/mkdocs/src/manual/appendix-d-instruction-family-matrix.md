# Appendix D. Instruction Family Matrix

## D.1 Scope

This appendix is generated from `docs/isa/manifest.yaml` and provides a source-synchronized matrix of PTO virtual instruction families.

## D.2 Coverage summary

| Category | Instruction Count |
|---|---:|
| Synchronization | 1 |
| Manual / Resource Binding | 4 |
| Elementwise (Tile-Tile) | 29 |
| Tile-Scalar / Tile-Immediate | 20 |
| Axis Reduce / Expand | 24 |
| Memory (GM <-> Tile) | 6 |
| Matrix Multiply | 8 |
| Data Movement / Layout | 13 |
| Complex | 15 |
| Communication | 11 |
| Total | 129 |

## D.3 Header synchronization status

- Header inventory source: `include/pto/common/pto_instr.hpp` (115 unique instruction APIs)
- Manifest inventory source: `docs/isa/manifest.yaml` (115 entries)
- Missing in manifest: none
- Present in manifest but missing in header: none

## D.4 Family matrix

| Category | Instruction | Diagram Template | Operand Contract | Semantic Page |
|---|---|---|---|---|
| Synchronization | [TSYNC](/docs/isa/tile/ops/sync-and-config/tsync.md) | `sync` | `producer, consumer` | `docs/isa/TSYNC.md` |
| Manual / Resource Binding | [TASSIGN](/docs/isa/tile/ops/sync-and-config/tassign.md) | `config` | `config, state` | `docs/isa/TASSIGN.md` |
| Manual / Resource Binding | [SETFMATRIX](/docs/isa/tile/ops/sync-and-config/setfmatrix.md) | `config` | `config, state` | `docs/isa/SETFMATRIX.md` |
| Elementwise (Tile-Tile) | [TADD](/docs/isa/tile/ops/elementwise-tile-tile/tadd.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADD.md` |
| Elementwise (Tile-Tile) | [TABS](/docs/isa/tile/ops/elementwise-tile-tile/tabs.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TABS.md` |
| Elementwise (Tile-Tile) | [TAND](/docs/isa/tile/ops/elementwise-tile-tile/tand.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TAND.md` |
| Elementwise (Tile-Tile) | [TOR](/docs/isa/tile/ops/elementwise-tile-tile/tor.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TOR.md` |
| Elementwise (Tile-Tile) | [TSUB](/docs/isa/tile/ops/elementwise-tile-tile/tsub.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUB.md` |
| Elementwise (Tile-Tile) | [TMUL](/docs/isa/tile/ops/elementwise-tile-tile/tmul.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMUL.md` |
| Elementwise (Tile-Tile) | [TMIN](/docs/isa/tile/ops/elementwise-tile-tile/tmin.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMIN.md` |
| Elementwise (Tile-Tile) | [TMAX](/docs/isa/tile/ops/elementwise-tile-tile/tmax.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMAX.md` |
| Elementwise (Tile-Tile) | [TCMP](/docs/isa/tile/ops/elementwise-tile-tile/tcmp.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCMP.md` |
| Elementwise (Tile-Tile) | [TDIV](/docs/isa/tile/ops/elementwise-tile-tile/tdiv.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TDIV.md` |
| Elementwise (Tile-Tile) | [TSHL](/docs/isa/tile/ops/elementwise-tile-tile/tshl.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHL.md` |
| Elementwise (Tile-Tile) | [TSHR](/docs/isa/tile/ops/elementwise-tile-tile/tshr.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHR.md` |
| Elementwise (Tile-Tile) | [TXOR](/docs/isa/tile/ops/elementwise-tile-tile/txor.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TXOR.md` |
| Elementwise (Tile-Tile) | [TLOG](/docs/isa/tile/ops/elementwise-tile-tile/tlog.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TLOG.md` |
| Elementwise (Tile-Tile) | [TRECIP](/docs/isa/tile/ops/elementwise-tile-tile/trecip.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRECIP.md` |
| Elementwise (Tile-Tile) | [TPRELU](/docs/isa/tile/ops/elementwise-tile-tile/tprelu.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TPRELU.md` |
| Elementwise (Tile-Tile) | [TADDC](/docs/isa/tile/ops/elementwise-tile-tile/taddc.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADDC.md` |
| Elementwise (Tile-Tile) | [TSUBC](/docs/isa/tile/ops/elementwise-tile-tile/tsubc.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUBC.md` |
| Elementwise (Tile-Tile) | [TCVT](/docs/isa/tile/ops/elementwise-tile-tile/tcvt.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCVT.md` |
| Elementwise (Tile-Tile) | [TSEL](/docs/isa/tile/ops/elementwise-tile-tile/tsel.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSEL.md` |
| Elementwise (Tile-Tile) | [TRSQRT](/docs/isa/tile/ops/elementwise-tile-tile/trsqrt.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRSQRT.md` |
| Elementwise (Tile-Tile) | [TSQRT](/docs/isa/tile/ops/elementwise-tile-tile/tsqrt.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSQRT.md` |
| Elementwise (Tile-Tile) | [TEXP](/docs/isa/tile/ops/elementwise-tile-tile/texp.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TEXP.md` |
| Elementwise (Tile-Tile) | [TNOT](/docs/isa/tile/ops/elementwise-tile-tile/tnot.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNOT.md` |
| Elementwise (Tile-Tile) | [TRELU](/docs/isa/tile/ops/elementwise-tile-tile/trelu.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRELU.md` |
| Elementwise (Tile-Tile) | [TNEG](/docs/isa/tile/ops/elementwise-tile-tile/tneg.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNEG.md` |
| Elementwise (Tile-Tile) | [TREM](/docs/isa/tile/ops/elementwise-tile-tile/trem.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TREM.md` |
| Elementwise (Tile-Tile) | [TFMOD](/docs/isa/tile/ops/elementwise-tile-tile/tfmod.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TFMOD.md` |
| Elementwise (Tile-Tile) | [TPOW](/docs/isa/tile/ops/elementwise-tile-tile/tpow.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TPOW.md` |
| Tile-Scalar / Tile-Immediate | [TEXPANDS](/docs/isa/tile/ops/tile-scalar-and-immediate/texpands.md) | `scalar` | `dst, src, scalar` | `docs/isa/TEXPANDS.md` |
| Tile-Scalar / Tile-Immediate | [TCMPS](/docs/isa/tile/ops/tile-scalar-and-immediate/tcmps.md) | `scalar` | `dst, src, scalar` | `docs/isa/TCMPS.md` |
| Tile-Scalar / Tile-Immediate | [TSELS](/docs/isa/tile/ops/tile-scalar-and-immediate/tsels.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSELS.md` |
| Tile-Scalar / Tile-Immediate | [TMINS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmins.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMINS.md` |
| Tile-Scalar / Tile-Immediate | [TADDS](/docs/isa/tile/ops/tile-scalar-and-immediate/tadds.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDS.md` |
| Tile-Scalar / Tile-Immediate | [TSUBS](/docs/isa/tile/ops/tile-scalar-and-immediate/tsubs.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBS.md` |
| Tile-Scalar / Tile-Immediate | [TDIVS](/docs/isa/tile/ops/tile-scalar-and-immediate/tdivs.md) | `scalar` | `dst, src, scalar` | `docs/isa/TDIVS.md` |
| Tile-Scalar / Tile-Immediate | [TMULS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmuls.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMULS.md` |
| Tile-Scalar / Tile-Immediate | [TFMODS](/docs/isa/tile/ops/tile-scalar-and-immediate/tfmods.md) | `scalar` | `dst, src, scalar` | `docs/isa/TFMODS.md` |
| Tile-Scalar / Tile-Immediate | [TREMS](/docs/isa/tile/ops/tile-scalar-and-immediate/trems.md) | `scalar` | `dst, src, scalar` | `docs/isa/TREMS.md` |
| Tile-Scalar / Tile-Immediate | [TMAXS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmaxs.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMAXS.md` |
| Tile-Scalar / Tile-Immediate | [TANDS](/docs/isa/tile/ops/tile-scalar-and-immediate/tands.md) | `scalar` | `dst, src, scalar` | `docs/isa/TANDS.md` |
| Tile-Scalar / Tile-Immediate | [TORS](/docs/isa/tile/ops/tile-scalar-and-immediate/tors.md) | `scalar` | `dst, src, scalar` | `docs/isa/TORS.md` |
| Tile-Scalar / Tile-Immediate | [TSHLS](/docs/isa/tile/ops/tile-scalar-and-immediate/tshls.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHLS.md` |
| Tile-Scalar / Tile-Immediate | [TSHRS](/docs/isa/tile/ops/tile-scalar-and-immediate/tshrs.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHRS.md` |
| Tile-Scalar / Tile-Immediate | [TXORS](/docs/isa/tile/ops/tile-scalar-and-immediate/txors.md) | `scalar` | `dst, src, scalar` | `docs/isa/TXORS.md` |
| Tile-Scalar / Tile-Immediate | [TLRELU](/docs/isa/tile/ops/tile-scalar-and-immediate/tlrelu.md) | `scalar` | `dst, src, scalar` | `docs/isa/TLRELU.md` |
| Tile-Scalar / Tile-Immediate | [TADDSC](/docs/isa/tile/ops/tile-scalar-and-immediate/taddsc.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDSC.md` |
| Tile-Scalar / Tile-Immediate | [TSUBSC](/docs/isa/tile/ops/tile-scalar-and-immediate/tsubsc.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBSC.md` |
| Tile-Scalar / Tile-Immediate | [TPOWS](/docs/isa/tile/ops/tile-scalar-and-immediate/tpows.md) | `scalar` | `dst, src, scalar` | `docs/isa/TPOWS.md` |
| Axis Reduce / Expand | [TROWSUM](/docs/isa/tile/ops/reduce-and-expand/trowsum.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWSUM.md` |
| Axis Reduce / Expand | [TROWPROD](/docs/isa/tile/ops/reduce-and-expand/trowprod.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWPROD.md` |
| Axis Reduce / Expand | [TCOLSUM](/docs/isa/tile/ops/reduce-and-expand/tcolsum.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLSUM.md` |
| Axis Reduce / Expand | [TCOLPROD](/docs/isa/tile/ops/reduce-and-expand/tcolprod.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLPROD.md` |
| Axis Reduce / Expand | [TCOLMAX](/docs/isa/tile/ops/reduce-and-expand/tcolmax.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMAX.md` |
| Axis Reduce / Expand | [TROWMAX](/docs/isa/tile/ops/reduce-and-expand/trowmax.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMAX.md` |
| Axis Reduce / Expand | [TROWMIN](/docs/isa/tile/ops/reduce-and-expand/trowmin.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMIN.md` |
| Axis Reduce / Expand | [TCOLARGMAX](/docs/isa/tile/ops/reduce-and-expand/tcolargmax.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLARGMAX.md` |
| Axis Reduce / Expand | [TCOLARGMIN](/docs/isa/tile/ops/reduce-and-expand/tcolargmin.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLARGMIN.md` |
| Axis Reduce / Expand | [TROWEXPAND](/docs/isa/tile/ops/reduce-and-expand/trowexpand.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPAND.md` |
| Axis Reduce / Expand | [TROWEXPANDDIV](/docs/isa/tile/ops/reduce-and-expand/trowexpanddiv.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDDIV.md` |
| Axis Reduce / Expand | [TROWEXPANDMUL](/docs/isa/tile/ops/reduce-and-expand/trowexpandmul.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMUL.md` |
| Axis Reduce / Expand | [TROWEXPANDSUB](/docs/isa/tile/ops/reduce-and-expand/trowexpandsub.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDSUB.md` |
| Axis Reduce / Expand | [TROWEXPANDADD](/docs/isa/tile/ops/reduce-and-expand/trowexpandadd.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDADD.md` |
| Axis Reduce / Expand | [TROWEXPANDMAX](/docs/isa/tile/ops/reduce-and-expand/trowexpandmax.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMAX.md` |
| Axis Reduce / Expand | [TROWEXPANDMIN](/docs/isa/tile/ops/reduce-and-expand/trowexpandmin.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMIN.md` |
| Axis Reduce / Expand | [TROWEXPANDEXPDIF](/docs/isa/tile/ops/reduce-and-expand/trowexpandexpdif.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDEXPDIF.md` |
| Axis Reduce / Expand | [TCOLMIN](/docs/isa/tile/ops/reduce-and-expand/tcolmin.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMIN.md` |
| Axis Reduce / Expand | [TCOLEXPAND](/docs/isa/tile/ops/reduce-and-expand/tcolexpand.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPAND.md` |
| Axis Reduce / Expand | [TCOLEXPANDDIV](/docs/isa/tile/ops/reduce-and-expand/tcolexpanddiv.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDDIV.md` |
| Axis Reduce / Expand | [TCOLEXPANDMUL](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmul.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMUL.md` |
| Axis Reduce / Expand | [TCOLEXPANDADD](/docs/isa/tile/ops/reduce-and-expand/tcolexpandadd.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDADD.md` |
| Axis Reduce / Expand | [TCOLEXPANDMAX](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmax.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMAX.md` |
| Axis Reduce / Expand | [TCOLEXPANDMIN](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmin.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMIN.md` |
| Axis Reduce / Expand | [TCOLEXPANDSUB](/docs/isa/tile/ops/reduce-and-expand/tcolexpandsub.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDSUB.md` |
| Axis Reduce / Expand | [TCOLEXPANDEXPDIF](/docs/isa/tile/ops/reduce-and-expand/tcolexpandexpdif.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDEXPDIF.md` |
| Memory (GM <-> Tile) | [TLOAD](/docs/isa/tile/ops/memory-and-data-movement/tload.md) | `memory` | `tile, global` | `docs/isa/TLOAD.md` |
| Memory (GM <-> Tile) | [TPREFETCH](/docs/isa/tile/ops/memory-and-data-movement/tprefetch.md) | `memory` | `tile, global` | `docs/isa/TPREFETCH.md` |
| Memory (GM <-> Tile) | [TSTORE](/docs/isa/tile/ops/memory-and-data-movement/tstore.md) | `memory` | `tile, global` | `docs/isa/TSTORE.md` |
| Memory (GM <-> Tile) | [TSTORE_FP](/docs/isa/tile/ops/memory-and-data-movement/tstore.md) | `memory` | `tile, global` | `docs/isa/TSTORE_FP.md` |
| Memory (GM <-> Tile) | [MGATHER](/docs/isa/tile/ops/memory-and-data-movement/mgather.md) | `memory` | `tile, global` | `docs/isa/MGATHER.md` |
| Memory (GM <-> Tile) | [MSCATTER](/docs/isa/tile/ops/memory-and-data-movement/mscatter.md) | `memory` | `tile, global` | `docs/isa/MSCATTER.md` |
| Matrix Multiply | [TGEMV_MX](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-mx.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_MX.md` |
| Matrix Multiply | [TMATMUL_MX](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-mx.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_MX.md` |
| Matrix Multiply | [TMATMUL](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL.md` |
| Matrix Multiply | [TMATMUL_ACC](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-acc.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_ACC.md` |
| Matrix Multiply | [TMATMUL_BIAS](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-bias.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_BIAS.md` |
| Matrix Multiply | [TGEMV](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV.md` |
| Matrix Multiply | [TGEMV_ACC](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-acc.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_ACC.md` |
| Matrix Multiply | [TGEMV_BIAS](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-bias.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_BIAS.md` |
| Data Movement / Layout | [TEXTRACT](/docs/isa/tile/ops/layout-and-rearrangement/textract.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT.md` |
| Data Movement / Layout | [TEXTRACT_FP](/docs/isa/tile/ops/layout-and-rearrangement/textract.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT_FP.md` |
| Data Movement / Layout | [TIMG2COL](/docs/isa/tile/ops/layout-and-rearrangement/timg2col.md) | `reshape_move` | `dst, src` | `docs/isa/TIMG2COL.md` |
| Data Movement / Layout | [TINSERT](/docs/isa/tile/ops/layout-and-rearrangement/tinsert.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT.md` |
| Data Movement / Layout | [TINSERT_FP](/docs/isa/tile/ops/layout-and-rearrangement/tinsert.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT_FP.md` |
| Data Movement / Layout | [TFILLPAD](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD.md` |
| Data Movement / Layout | [TFILLPAD_INPLACE](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad-inplace.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_INPLACE.md` |
| Data Movement / Layout | [TFILLPAD_EXPAND](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad-expand.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_EXPAND.md` |
| Data Movement / Layout | [TMOV](/docs/isa/tile/ops/layout-and-rearrangement/tmov.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV.md` |
| Data Movement / Layout | [TMOV_FP](/docs/isa/tile/ops/layout-and-rearrangement/tmov.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV_FP.md` |
| Data Movement / Layout | [TRESHAPE](/docs/isa/tile/ops/layout-and-rearrangement/treshape.md) | `reshape_move` | `dst, src` | `docs/isa/TRESHAPE.md` |
| Data Movement / Layout | [TTRANS](/docs/isa/tile/ops/layout-and-rearrangement/ttrans.md) | `reshape_move` | `dst, src` | `docs/isa/TTRANS.md` |
| Data Movement / Layout | [TSUBVIEW](/docs/isa/tile/ops/sync-and-config/subview.md) | `reshape_move` | `dst, src, rowOffset, colOffset` | `docs/isa/TSUBVIEW.md` |
| Data Movement / Layout | [TGET_SCALE_ADDR](/docs/isa/tile/ops/sync-and-config/get-scale-addr.md) | `reshape_move` | `dst, src` | `docs/isa/TGET_SCALE_ADDR.md` |
| Data Movement / Layout | [TCONCAT](/docs/isa/tile/ops/layout-and-rearrangement/tconcat.md) | `reshape_move` | `dst, src0, src1` | `docs/isa/TCONCAT.md` |
| Complex | [TPRINT](/docs/isa/tile/ops/irregular-and-complex/tprint.md) | `complex` | `dst, src0, src1` | `docs/isa/TPRINT.md` |
| Complex | [TMRGSORT](/docs/isa/tile/ops/irregular-and-complex/tmrgsort.md) | `complex` | `dst, src0, src1` | `docs/isa/TMRGSORT.md` |
| Complex | [TSORT32](/docs/isa/tile/ops/irregular-and-complex/tsort32.md) | `complex` | `dst, src0, src1` | `docs/isa/TSORT32.md` |
| Complex | [TGATHER](/docs/isa/tile/ops/irregular-and-complex/tgather.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHER.md` |
| Complex | [TCI](/docs/isa/tile/ops/irregular-and-complex/tci.md) | `complex` | `dst, src0, src1` | `docs/isa/TCI.md` |
| Complex | [TTRI](/docs/isa/tile/ops/irregular-and-complex/ttri.md) | `complex` | `dst, src0, src1` | `docs/isa/TTRI.md` |
| Complex | [TPARTADD](/docs/isa/tile/ops/irregular-and-complex/tpartadd.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTADD.md` |
| Complex | [TPARTMUL](/docs/isa/tile/ops/irregular-and-complex/tpartmul.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMUL.md` |
| Complex | [TPARTMAX](/docs/isa/tile/ops/irregular-and-complex/tpartmax.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMAX.md` |
| Complex | [TPARTMIN](/docs/isa/tile/ops/irregular-and-complex/tpartmin.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMIN.md` |
| Complex | [TPARTARGMAX](/docs/isa/TPARTARGMAX.md) | `complex` | `dstVal, dstIdx, src0Val, src1Val, src0Idx, src1Idx` | `docs/isa/TPARTARGMAX.md` |
| Complex | [TPARTARGMIN](/docs/isa/TPARTARGMIN.md) | `complex` | `dstVal, dstIdx, src0Val, src1Val, src0Idx, src1Idx` | `docs/isa/TPARTARGMIN.md` |
| Complex | [TGATHERB](/docs/isa/tile/ops/irregular-and-complex/tgatherb.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHERB.md` |
| Complex | [TSCATTER](/docs/isa/tile/ops/irregular-and-complex/tscatter.md) | `complex` | `dst, src0, src1` | `docs/isa/TSCATTER.md` |
| Complex | [TQUANT](/docs/isa/tile/ops/irregular-and-complex/tquant.md) | `complex` | `dst, src0, src1` | `docs/isa/TQUANT.md` |
| Communication | [TPUT](/docs/isa/comm/TPUT.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TPUT.md` |
| Communication | [TGET](/docs/isa/comm/TGET.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TGET.md` |
| Communication | [TPUT_ASYNC](/docs/isa/comm/TPUT_ASYNC.md) | `comm` | `dst, src, session` | `docs/isa/comm/TPUT_ASYNC.md` |
| Communication | [TGET_ASYNC](/docs/isa/comm/TGET_ASYNC.md) | `comm` | `dst, src, session` | `docs/isa/comm/TGET_ASYNC.md` |
| Communication | [TNOTIFY](/docs/isa/comm/TNOTIFY.md) | `comm` | `signal, value, op` | `docs/isa/comm/TNOTIFY.md` |
| Communication | [TWAIT](/docs/isa/comm/TWAIT.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TWAIT.md` |
| Communication | [TTEST](/docs/isa/comm/TTEST.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TTEST.md` |
| Communication | [TGATHER](/docs/isa/comm/TGATHER.md) | `comm` | `group, dst, staging` | `docs/isa/comm/TGATHER.md` |
| Communication | [TSCATTER](/docs/isa/comm/TSCATTER.md) | `comm` | `group, src, staging` | `docs/isa/comm/TSCATTER.md` |
| Communication | [TREDUCE](/docs/isa/comm/TREDUCE.md) | `comm` | `group, dst, acc, recv` | `docs/isa/comm/TREDUCE.md` |
| Communication | [TBROADCAST](/docs/isa/comm/TBROADCAST.md) | `comm` | `group, src, staging` | `docs/isa/comm/TBROADCAST.md` |

## D.5 Notes

- Per-instruction semantics remain canonical in `docs/isa/*.md`.
- This appendix is a taxonomy and coverage matrix, not a replacement for per-op normative semantics.
