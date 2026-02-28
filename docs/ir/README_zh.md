# PTO IR 鎿嶄綔鍙傝€?
鏈洰褰曞寘鍚?PTO IR 鎿嶄綔鐨勫叏闈㈡枃妗ｏ紝娑电洊 ISA 绾у埆鐨?tile 鎿嶄綔鍜?PTO Level-1 鍙?Level-2 涓棿琛ㄧず涓娇鐢ㄧ殑杈呭姪 IR 鏋勯€犮€?
---

## 姒傝堪

PTO AS 鎻愪緵 **116 涓?tile 鎿嶄綔**銆?*11 涓緟鍔╁嚱鏁?*銆?*47 涓爣閲忕畻鏈搷浣?*鍜?**7 涓帶鍒舵祦鎿嶄綔**銆?
姣忎釜鎿嶄綔閮借褰曟湁锛?- **IR Level 1 (SSA)**锛氶潤鎬佸崟璧嬪€煎舰寮?- **IR Level 2 (DPS)**锛氱洰鏍囦紶閫掗鏍?- **鏁板璇箟**锛氬舰寮忓寲鏁板瑙ｉ噴
- **绾︽潫**锛氱被鍨嬨€佸竷灞€鍜岃繍琛屾椂瑕佹眰

---

## 杈呭姪鍑芥暟锛?1 涓嚱鏁帮級

**鏂囨。**锛歔杈呭姪鍑芥暟](PTO-IR-ops_zh.md)

鐢ㄤ簬寮犻噺瑙嗗浘绠＄悊銆乼ile 鍒嗛厤鍜屽悓姝ョ殑 IR 绾ф瀯閫狅細

- **寮犻噺瑙嗗浘**锛歚make_tensor_view`銆乣partition_view`
- **Tile 绠＄悊**锛歚alloc_tile`銆乣tgetval`銆乣tsetval`
- **绱㈠紩**锛歚get_block_idx`銆乣get_subblock_idx`銆乣get_block_num`銆乣get_subblock_num`
- **鎸囬拡杩愮畻**锛歚addptr`
- **鍚屾**锛歚record_event`銆乣wait_event`銆乣barrier`銆乣PIPE_BARRIER`

---

## Tile 鎿嶄綔锛?16 涓搷浣滐級

### 閫愬厓绱犳搷浣滐紙Tile-Tile锛? 28 涓搷浣?**鏂囨。**锛歔閫愬厓绱犳搷浣淽(PTO-IR-elementwise-ops_zh.md)

- **绠楁湳**锛歚TADD`銆乣TSUB`銆乣TMUL`銆乣TDIV`銆乣TABS`銆乣TNEG`
- **浣嶈繍绠?*锛歚TAND`銆乣TOR`銆乣TXOR`銆乣TNOT`銆乣TSHL`銆乣TSHR`
- **姣旇緝**锛歚TCMP`銆乣TMIN`銆乣TMAX`
- **鏁板**锛歚TLOG`銆乣TEXP`銆乣TSQRT`銆乣TRSQRT`銆乣TRECIP`
- **婵€娲?*锛歚TRELU`銆乣TPRELU`
- **绫诲瀷杞崲**锛歚TCVT`
- **鏉′欢**锛歚TSEL`
- **澶嶅悎**锛歚TADDC`銆乣TSUBC`
- **鍙栨ā**锛歚TREM`銆乣TFMOD`

### Tile-鏍囬噺鎿嶄綔 - 19 涓搷浣?**鏂囨。**锛歔Tile-鏍囬噺鎿嶄綔](PTO-IR-tile-scalar-ops_zh.md)

- **绠楁湳**锛歚TADDS`銆乣TSUBS`銆乣TMULS`銆乣TDIVS`銆乣TMINS`銆乣TMAXS`
- **浣嶈繍绠?*锛歚TANDS`銆乣TORS`銆乣TXORS`銆乣TSHLS`銆乣TSHRS`
- **鍙栨ā**锛歚TREMS`銆乣TFMODS`
- **骞挎挱**锛歚TEXPANDS`
- **姣旇緝**锛歚TCMPS`
- **鏉′欢**锛歚TSELS`
- **婵€娲?*锛歚TLRELU`
- **澶嶅悎**锛歚TADDSC`銆乣TSUBSC`

### 杞村綊绾﹀拰鎵╁睍 - 23 涓搷浣?**鏂囨。**锛歔杞村綊绾﹀拰鎵╁睍](PTO-IR-axis-ops_zh.md)

- **琛屽綊绾?*锛歚TROWSUM`銆乣TROWMAX`銆乣TROWMIN`
- **鍒楀綊绾?*锛歚TCOLSUM`銆乣TCOLMAX`銆乣TCOLMIN`銆乣TCOLPROD`
- **琛屾墿灞?*锛歚TROWEXPAND`銆乣TROWEXPANDADD`銆乣TROWEXPANDMUL`銆乣TROWEXPANDDIV`銆乣TROWEXPANDSUB`銆乣TROWEXPANDMAX`銆乣TROWEXPANDMIN`銆乣TROWEXPANDEXPDIF`
- **鍒楁墿灞?*锛歚TCOLEXPAND`銆乣TCOLEXPANDADD`銆乣TCOLEXPANDMUL`銆乣TCOLEXPANDDIV`銆乣TCOLEXPANDSUB`銆乣TCOLEXPANDMAX`銆乣TCOLEXPANDMIN`銆乣TCOLEXPANDEXPDIF`

### 鍐呭瓨鎿嶄綔 - 6 涓搷浣?**鏂囨。**锛歔鍐呭瓨鎿嶄綔](PTO-IR-memory-ops_zh.md)

- **鍔犺浇/瀛樺偍**锛歚TLOAD`銆乣TSTORE`銆乣TSTORE_FP`銆乣TPREFETCH`
- **鏀堕泦/鍒嗘暎**锛歚MGATHER`銆乣MSCATTER`

### 鐭╅樀涔樻硶 - 8 涓搷浣?**鏂囨。**锛歔鐭╅樀涔樻硶](PTO-IR-matrix-ops_zh.md)

- **鍩虹**锛歚TMATMUL`銆乣TMATMUL_ACC`銆乣TMATMUL_BIAS`
- **娣峰悎绮惧害**锛歚TMATMUL_MX`
- **鍚戦噺**锛歚TGEMV`銆乣TGEMV_ACC`銆乣TGEMV_BIAS`銆乣TGEMV_MX`

### 鏁版嵁绉诲姩鍜屽竷灞€ - 12 涓搷浣?**鏂囨。**锛歔鏁版嵁绉诲姩鍜屽竷灞€](PTO-IR-data-movement-ops_zh.md)

- **鎻愬彇/鎻掑叆**锛歚TEXTRACT`銆乣TEXTRACT_FP`銆乣TINSERT`銆乣TINSERT_FP`
- **杞崲**锛歚TTRANS`銆乣TRESHAPE`銆乣TIMG2COL`
- **绉诲姩**锛歚TMOV`銆乣TMOV_FP`
- **濉厖**锛歚TFILLPAD`銆乣TFILLPAD_INPLACE`銆乣TFILLPAD_EXPAND`

### 澶嶆潅鎿嶄綔 - 13 涓搷浣?**鏂囨。**锛歔澶嶆潅鎿嶄綔](PTO-IR-complex-ops_zh.md)

- **鎺掑簭**锛歚TSORT32`銆乣TMRGSORT`
- **鏀堕泦**锛歚TGATHER`銆乣TGATHERB`銆乣TSCATTER`
- **閮ㄥ垎鎿嶄綔**锛歚TPARTADD`銆乣TPARTMUL`銆乣TPARTMAX`銆乣TPARTMIN`
- **瀹炵敤宸ュ叿**锛歚TCI`銆乣TTRI`銆乣TQUANT`銆乣TPRINT`

### 鎵嬪姩璧勬簮缁戝畾 - 6 涓搷浣?**鏂囨。**锛歔鎵嬪姩璧勬簮缁戝畾](PTO-IR-manual-binding-ops_zh.md)

- **璧嬪€?*锛歚TASSIGN`
- **妯″紡閰嶇疆**锛歚TSETHF32MODE`銆乣TSETTF32MODE`銆乣TSETFMATRIX`
- **IMG2COL 閰嶇疆**锛歚TSET_IMG2COL_RPT`銆乣TSET_IMG2COL_PADDING`

---

## 鏍囬噺绠楁湳鎿嶄綔锛?7 涓搷浣滐級

**鏂囨。**锛歔鏍囬噺绠楁湳鎿嶄綔](PTO-IR-scalar-arith-ops_zh.md)

鏉ヨ嚜 MLIR `arith` 鏂硅█鐨勬爣鍑嗘爣閲忔搷浣滐紙浠呮爣閲忥紝鏃犲悜閲?寮犻噺锛夛細

- **鏁存暟绠楁湳**锛歚addi`銆乣subi`銆乣muli`銆乣divsi`銆乣divui`銆乣remsi`銆乣remui`銆乣ceildivsi`銆乣ceildivui`銆乣floordivsi`
- **娴偣绠楁湳**锛歚addf`銆乣subf`銆乣mulf`銆乣divf`銆乣remf`銆乣negf`
- **浣嶈繍绠?*锛歚andi`銆乣ori`銆乣xori`
- **绉讳綅**锛歚shli`銆乣shrsi`銆乣shrui`
- **姣旇緝**锛歚cmpi`銆乣cmpf`
- **鏈€灏?鏈€澶?*锛歚minsi`銆乣minui`銆乣maxsi`銆乣maxui`銆乣minimumf`銆乣maximumf`銆乣minnumf`銆乣maxnumf`
- **绫诲瀷杞崲**锛歚extsi`銆乣extui`銆乣trunci`銆乣extf`銆乣truncf`銆乣sitofp`銆乣uitofp`銆乣fptosi`銆乣fptoui`銆乣bitcast`銆乣index_cast`銆乣index_castui`
- **鐗规畩鎿嶄綔**锛歚select`銆乣constant`
- **鎵╁睍绠楁湳**锛歚addui_extended`銆乣mulsi_extended`銆乣mului_extended`

---

## 鎺у埗娴佹搷浣滐紙7 涓搷浣滐級

**鏂囨。**锛歔鎺у埗娴佹搷浣淽(PTO-IR-control-flow-ops_zh.md)

鏉ヨ嚜 MLIR `scf` 鏂硅█鐨勭粨鏋勫寲鎺у埗娴佹搷浣滐細

- **寰幆**锛歚scf.for`銆乣scf.while`
- **鏉′欢**锛歚scf.if`銆乣scf.index_switch`
- **鍖哄煙**锛歚scf.execute_region`
- **缁堟绗?*锛歚scf.yield`銆乣scf.condition`

---

## 鐩稿叧璧勬簮

- [**ISA 鎸囦护鍙傝€?*](../isa/README_zh.md)锛氶€愭潯鎸囦护鐨勮鑼冭涔?- [**PTO-AS 鏂囨硶**](../grammar/PTO-AS_zh.md)锛氭眹缂栬瑷€璇硶鍜屾枃娉?