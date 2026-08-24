/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

c0 = const(0)
c1 = const(1)
c_s0 = const(S0)
c_head = const(HEAD)
c_cube_s1 = const(CUBE_S1)
c_s1_tile = const(S1_TILE)
c_preload = const(QK_PRELOAD)
s0 = s.index_cast(s0_i64)
s1 = s.index_cast(s1_i64)
num_tiles_s1 = s1 // c_s1_tile
steady_tiles = num_tiles_s1 - c_preload

bid, qb_start, qb_end = compute_qb_range(c1)

# The P slot is fp16-typed, so address it via the fp16-cast slot buffer.
# GM_P_OFF_F32 is in fp32 elements; double for fp16 element stride.
gm_slots = block_gm_slots(gm_slot_buffer, gm_slot_buffer_fp16, bid)

# ---- QK pipe (cube producer): l2g2l GM-staged slot ----
qk_slot_view = as_tensor(qk_slot_ty, ptr=gm_slots["qk"], shape=[c_s0, c_s1_tile], strides=[c_s1_tile, c1])
qk_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=1, slot_size=SLOT_SIZE_QK, slot_num=SLOT_NUM, gm_addr=qk_slot_view, flag_base=0
)

# ---- PV pipe (cube producer): l2g2l GM-staged slot ----
pv_slot_view = as_tensor(pv_slot_ty, ptr=gm_slots["pv"], shape=[c_s0, c_head], strides=[c_head, c1])
pv_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=1, slot_size=SLOT_SIZE_PV, slot_num=SLOT_NUM, gm_addr=pv_slot_view, flag_base=4
)

# ---- P pipe (cube consumer of vec output): l2g2l GM-staged slot ----
p_slot_view_cube = as_tensor(
    p_cube_slot_ty, ptr=gm_slots["p"], shape=[c_s0, c_s1_tile], strides=[c_s1_tile, c1]
)
p_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=2, slot_size=SLOT_SIZE_P, slot_num=SLOT_NUM, gm_addr=p_slot_view_cube, flag_base=2
)

# ---- Allocate cube tiles. Match the manual kernel's ping-pong for
# K/P/V MAT tiles where L1 capacity allows it. RIGHT is single-buffered
# because two 128x128 RIGHT tiles for both QK and PV overflow L0B.
q_mat = pto.alloc_tile(q_mat_ty)
q_left = pto.alloc_tile(q_left_ty)
k_mat_a = pto.alloc_tile(k_mat_ty)
k_mat_b = pto.alloc_tile(k_mat_ty)
k_right_a = pto.alloc_tile(k_right_ty)
qk_acc_a = pto.alloc_tile(qk_acc_ty)
p_recv_a = pto.alloc_tile(p_recv_ty)
p_left_a = pto.alloc_tile(p_left_ty)
v_mat_a = pto.alloc_tile(v_mat_ty)
v_right_a = pto.alloc_tile(v_right_ty)
pv_acc_a = pto.alloc_tile(pv_acc_ty)
k_mat = [k_mat_a, k_mat_b]
k_right = [k_right_a, k_right_a]
qk_acc = [qk_acc_a, qk_acc_a]
p_recv = [p_recv_a, p_recv_a]
p_left = [p_left_a, p_left_a]
v_mat = [v_mat_a, v_mat_a]
v_right = [v_right_a, v_right_a]
pv_acc = [pv_acc_a, pv_acc_a]

tv_q = as_tensor(qkv_tensor_ty, ptr=gm_q, shape=[s0, c_head], strides=[c_head, c1])
tv_k = as_tensor(qkv_tensor_ty, ptr=gm_k, shape=[c_head, s1], strides=[c1, c_head], layout="DN")
tv_v = as_tensor(qkv_tensor_ty, ptr=gm_v, shape=[s1, c_head], strides=[c_head, c1])

qk_entry = pto.declare_global(qk_slot_ty)
p_entry = pto.declare_global(p_cube_slot_ty)
pv_entry = pto.declare_global(pv_slot_ty)

# Closures over the shared tile state. The steady state overlaps PV for
# the current S1 tile with QK for the next S1 tile at CUBE_S1 granularity.
def emit_qk_sub(s1_tile_idx, sub, b):
    kt_view = slice_view(
        kt_sub_ty,
        source=tv_k,
        offsets=[c0, s1_tile_idx * c_s1_tile + const(sub * CUBE_S1)],
        sizes=[c_head, c_cube_s1],
    )
    pto.load(kt_view, k_mat[b])
    tile.mov(k_mat[b], k_right[b])
    tile.matmul(q_left, k_right[b], qk_acc[b])
    slot_part = slice_view(
        qk_slot_part_ty, source=qk_entry, offsets=[c0, const(sub * CUBE_S1)], sizes=[c_s0, c_cube_s1]
    )
    pto.store(qk_acc[b], slot_part)

def emit_qk(s1_tile_idx, b):
    pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
    for sub in range(TILE_FACTOR):
        emit_qk_sub(s1_tile_idx, sub, b)
    pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

def emit_pv_sub(t_idx, sub, b):
    p_part = slice_view(
        p_cube_slot_part_ty, source=p_entry, offsets=[c0, const(sub * CUBE_S1)], sizes=[c_s0, c_cube_s1]
    )
    pto.load(p_part, p_recv[b])
    tile.mov(p_recv[b], p_left[b])
    v_view = slice_view(
        v_sub_ty, source=tv_v, offsets=[t_idx * c_s1_tile + const(sub * CUBE_S1), c0], sizes=[c_cube_s1, c_head]
    )
    pto.load(v_view, v_mat[b])
    tile.mov(v_mat[b], v_right[b])
    if sub == 0:
        tile.matmul(p_left[b], v_right[b], pv_acc[b])
    else:
        tile.matmul_acc(pv_acc[b], p_left[b], v_right[b], pv_acc[b])

def push_pv(b):
    pto.tfree(p_pipe, SPLIT_UP_DOWN, entry=p_entry)
    pto.talloc(pv_entry, pv_pipe, SPLIT_UP_DOWN)
    pv_part = slice_view(pv_slot_part_ty, source=pv_entry, offsets=[c0, c0], sizes=[c_s0, c_head])
    pto.store(pv_acc[b], pv_part)
    pto.tpush(pv_entry, pv_pipe, SPLIT_UP_DOWN)

def emit_pv(t_idx, b):
    pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
    for sub in range(TILE_FACTOR):
        emit_pv_sub(t_idx, sub, b)
    push_pv(b)

def emit_qk_pv_interleaved(next_idx, current_idx, b):
    pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
    for sub in range(TILE_FACTOR):
        emit_pv_sub(current_idx, sub, b)
        if sub == 0:
            pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
        if sub == TILE_FACTOR - 1:
            push_pv(b)
        emit_qk_sub(next_idx, sub, b)
        if sub == TILE_FACTOR - 1:
            pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

# ---- Q-block loop ----
if KV_SPLIT == 1:
    for qb in pto.range(qb_start, qb_end, c1):
        q_view = slice_view(q_sub_ty, source=tv_q, offsets=[qb * c_s0, c0], sizes=[c_s0, c_head])
        pto.load(q_view, q_mat)
        tile.mov(q_mat, q_left)

        # ---- prologue: emit QK[0..QK_PRELOAD-1] ---------------------
        # V loading is now inline in emit_pv (per-sub-tile), so no preload.
        for kp in range(QK_PRELOAD):
            emit_qk(const(kp), kp % 2)

        # ---- steady state -------------------------------------------
        # Match the 140tflops schedule: consume current P/PV and emit the
        # next QK slot at CUBE_S1 sub-tile granularity.
        for tile_id in pto.range(c0, steady_tiles, c1):
            next_tile = tile_id + c_preload
            emit_qk_pv_interleaved(next_tile, tile_id, 0)

        # ---- epilogue: drain the last QK_PRELOAD PVs ----------------
        for k in range(QK_PRELOAD):
            emit_pv(steady_tiles + const(k), 0)
else:
    # ---- T1-A split-KV: one work-unit = (q block, kv chunk) ----
    # Tile indices passed to emit_* are ABSOLUTE into K/V, so offset
    # every local tile by kv_base = chunk * tiles_per_chunk. The FIFO
    # slot ring and accumulators are per-core, unchanged.
    c_kv_split = const(KV_SPLIT)
    tiles_per_chunk = num_tiles_s1 // c_kv_split
    steady_chunk = tiles_per_chunk - c_preload
    for u in pto.range(qb_start, qb_end, c1):
        qb = u // c_kv_split
        chunk = u % c_kv_split
        kv_base = chunk * tiles_per_chunk
        q_view = slice_view(q_sub_ty, source=tv_q, offsets=[qb * c_s0, c0], sizes=[c_s0, c_head])
        pto.load(q_view, q_mat)
        tile.mov(q_mat, q_left)

        for kp in range(QK_PRELOAD):
            emit_qk(kv_base + const(kp), kp % 2)

        for tile_id in pto.range(c0, steady_chunk, c1):
            local = kv_base + tile_id
            emit_qk_pv_interleaved(local + c_preload, local, 0)

        for k in range(QK_PRELOAD):
            emit_pv(kv_base + steady_chunk + const(k), 0)
