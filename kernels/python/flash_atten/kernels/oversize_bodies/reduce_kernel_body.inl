c0 = const(0)
c1 = const(1)
c_s0 = const(S0)
c_s0_half = const(S0_HALF)
c_vec_s0 = const(Vec_S0)
c_head = const(HEAD)
c_kv_split = const(KV_SPLIT)
s0 = s.index_cast(s0_i64)
scale = const(1.0 / math.sqrt(HEAD), s.float32)

partial_views = partial_tensor_views(gm_slot_buffer, c1, c_head)
tv_o_parts = partial_views["o"]
tv_m_parts = partial_views["m"]
tv_l_parts = partial_views["l"]
tv_o = as_tensor(o_tensor_ty, ptr=gm_o, shape=[s0, c_head], strides=[c_head, c1])

o_acc = pto.alloc_tile(o_vec_ty)
o_i = pto.alloc_tile(o_vec_ty)
gmax = pto.alloc_tile(red_ty)
mtmp = pto.alloc_tile(red_ty)
gsum = pto.alloc_tile(red_ty)
ltmp = pto.alloc_tile(red_ty)
corr = pto.alloc_tile(red_ty)
reduce_tiles = {
    "o_acc": o_acc,
    "o_i": o_i,
    "gmax": gmax,
    "mtmp": mtmp,
    "gsum": gsum,
    "ltmp": ltmp,
    "corr": corr,
}
reduce_ctx = {
    "tv_o": tv_o,
    "tv_o_parts": tv_o_parts,
    "tv_m_parts": tv_m_parts,
    "tv_l_parts": tv_l_parts,
    "c0": c0,
    "c1": c1,
    "c_s0": c_s0,
    "c_vec_s0": c_vec_s0,
    "c_head": c_head,
    "scale": scale,
    "tiles": reduce_tiles,
}

sb_idx = s.index_cast(pto.get_subblock_idx())
row_off_sb = sb_idx * c_s0_half
bid, qb_start, qb_end = compute_range(NUM_Q_BLOCKS, c1)

for qb in pto.range(qb_start, qb_end, c1):
    u0 = qb * c_kv_split
    for row_slice in range(TILE_FACTOR):
        row = row_off_sb + const(row_slice * Vec_S0)
        reduce_row_slice(reduce_ctx, qb, u0, row)
