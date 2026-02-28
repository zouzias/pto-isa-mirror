# PTO IR 闈?ISA 杩愮畻锛圠evel-1 / Level-2锛?

## 1. 鑼冨洿

鏈〉缁欏嚭 `~/pto-isa.txt` 涓潪 ISA 鎸囦护鏉＄洰鐨?PTO IR 杩愮畻瑙勮寖銆?

- Level-1锛歋SA 褰㈡€侊紝鐢辩紪璇戝櫒绠＄悊鍒嗛厤涓庡悓姝ャ€?
- Level-2锛欴PS 褰㈡€侊紝鏀寔鏄惧紡缂撳啿澶嶇敤涓庡悓姝ュ師璇€?

## 2. View 杩愮畻

### 2.1 `make_tensor_view`

```text
// L1
%dst = pto.make_tensor_view %ptr, shape = [sh1,sh2,sh3,sh4,sh5] strides = [st1,st2,st3,st4,st5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

### 2.2 `partition_view`

```text
// L1
%dst = pto.partition_view %src, offsets = [of1,of2,of3,of4,of5], sizes = [sh1,sh2,sh3,sh4,sh5] : !pto.tensor_view<sh1xsh2xsh3xsh4xsh5xdtype> -> !pto.partition_tensor_view<sh1xsh2xsh3xsh4xsh5xdtype>
```

## 3. Tile 鍒嗛厤

### 3.1 `alloc_tile`锛堥潤鎬佸弬鏁帮級

```text
// L2
%dst = pto.alloc_tile : !pto.tile_buf<loc, dtype, rows, cols, v_row, v_col, blayout, slayou, fractal, pad>
```

### 3.2 `alloc_tile`锛堝姩鎬佹湁鏁堝煙锛?

```text
// L2
%dst = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc, dtype, rows, cols, v_row=?, v_col=?, blayout, slayou, fractal, pad>
```

## 4. 鏍稿弬鏁版煡璇?

### 4.1 `get_block_idx`

```text
// L1 / L2
%idx = pto.get_block_idx
```

### 4.2 `get_subblock_idx`

```text
// L1 / L2
%idx = pto.get_subblock_idx
```

### 4.3 `get_block_num`

```text
// L1 / L2
%num = pto.get_block_num
```

### 4.4 `get_subblock_num`

```text
// L1 / L2
%num = pto.get_subblock_num
```

## 5. 鎸囬拡涓庢爣閲忚闂?

### 5.1 `addptr`

```text
// L2
%ptr_new = pto.addptr %ptr, %offset
```

### 5.2 `tgetval`

```text
// L2
pto.tgetval ins(%src, %index : !pto.tile_buf<...>, dtype) outs(%val : dtype)
```

### 5.3 `tsetval`

```text
// L2
pto.tsetval ins(%index, %val : dtype, dtype) outs(%dst : !pto.tile_buf<...>)
```

## 6. 鍚屾鍘熻锛圠evel-2锛?

### 6.1 `record_event`

```text
pto.record_event[src_op, dst_op, eventID]
```

褰撳墠琛ㄦ牸鏀寔 op锛歚TLOAD`銆乣TSTORE_ACC`銆乣TSTORE_VEC`銆乣TMOV_M2L`銆乣TMOV_M2S`銆乣TMOV_M2B`銆乣TMOV_M2V`銆乣TMOV_V2M`銆乣TMATMUL`銆乣TVEC`銆?

### 6.2 `wait_event`

```text
pto.wait_event[src_op, dst_op, eventID]
```

褰撳墠琛ㄦ牸鏀寔 op锛歚TLOAD`銆乣TSTORE_ACC`銆乣TSTORE_VEC`銆乣TMOV_M2L`銆乣TMOV_M2S`銆乣TMOV_M2B`銆乣TMOV_M2V`銆乣TMOV_V2M`銆乣TMATMUL`銆乣TVEC`銆?

### 6.3 `barrier`

```text
pto.barrier(op)
```

褰撳墠琛ㄦ牸鏀寔 op锛歚TVEC`銆乣TMATMUL`銆?

## 7. 涓€鑷存€ц鏄?

- 杩欎簺闈?ISA PTO IR 杩愮畻缁熶竴鏀舵暃鍒版湰鑺傛枃妗ｏ紝涓嶈繘鍏?`docs/isa/` 鐨?manifest 椹卞姩鎸囦护绱㈠紩銆?
- `docs/isa/TSYNC.md` / `docs/isa/TSYNC_zh.md` 浠嶆槸 ISA 灞傚悓姝ヨ涔夋潈濞佹潵婧愩€?
- 褰?`~/pto-isa.txt` 鍙樻洿鏃讹紝鏈〉搴斿湪鍚屼竴鍙樻洿闆嗕腑鍚屾鏇存柊銆?
