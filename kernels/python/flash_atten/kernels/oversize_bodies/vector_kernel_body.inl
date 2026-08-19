c0 = const(0)
c1 = const(1)
c2 = const(2)
c_s0 = const(S0)
c_s0_half = const(S0_HALF)
c_vec_gu_rows = const(VecGuRows)
c_vec_s0 = const(Vec_S0)
c_head = const(HEAD)
c_s1_tile = const(S1_TILE)
c_preload = const(QK_PRELOAD)
s0 = s.index_cast(s0_i64)
s1 = s.index_cast(s1_i64)
num_tiles_s1 = s1 // c_s1_tile
steady_tiles = num_tiles_s1 - c_preload

bid, qb_start, qb_end = compute_qb_range(c1)

gm_slots = block_gm_slots(gm_slot_buffer, gm_slot_buffer_fp16, bid)

# ---- QK pipe (vec consumer): l2g2l GM-staged slot ----
# Vec sees one slot as [VecGuRows, S1_TILE] -- SPLIT_UP_DOWN halves
# the row count when crossing into the subblock; per row_slice we
# tload a [Vec_S0, S1_TILE] partition.
qk_slot_view = as_tensor(
    qk_vec_slot_ty, ptr=gm_slots["qk"], shape=[c_vec_gu_rows, c_s1_tile], strides=[c_s1_tile, c1]
)
qk_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=1, slot_size=SLOT_SIZE_QK, slot_num=SLOT_NUM, gm_addr=qk_slot_view, flag_base=0
)
# ---- PV pipe (vec consumer): l2g2l GM-staged slot ----
pv_slot_view = as_tensor(
    pv_vec_slot_ty, ptr=gm_slots["pv"], shape=[c_vec_gu_rows, c_head], strides=[c_head, c1]
)
pv_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=1, slot_size=SLOT_SIZE_PV, slot_num=SLOT_NUM, gm_addr=pv_slot_view, flag_base=4
)

# ---- P pipe (vec producer): l2g2l GM-staged slot ----
p_slot_view = as_tensor(p_slot_ty, ptr=gm_slots["p"], shape=[c_vec_gu_rows, c_s1_tile], strides=[c_s1_tile, c1])
p_pipe = pto.initialize_l2g2l_pipe(
    dir_mask=2, slot_size=SLOT_SIZE_P, slot_num=SLOT_NUM, gm_addr=p_slot_view, flag_base=2
)

qk_vec = pto.alloc_tile(qk_vec_ty)  # [Vec_S0, S1_TILE] working
tmp = pto.alloc_tile(qk_vec_ty)  # [Vec_S0, S1_TILE] row-reduce scratch
p_fp32 = pto.alloc_tile(p_fp32_ty)
p_fp16 = pto.alloc_tile(p_fp16_ty)
pv_vec = [pto.alloc_tile(pv_vec_ty) for _ in range(TILE_FACTOR)]
o_tile = [pto.alloc_tile(o_vec_ty) for _ in range(TILE_FACTOR)]
running_max = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
running_sum = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
local_max = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
local_sum = [pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)]
exp_max_ring = [[pto.alloc_tile(red_ty) for _ in range(TILE_FACTOR)] for _ in range(EXP_RING)]

scale = const(1.0 / math.sqrt(HEAD), s.float32)
c_exp_ring = const(EXP_RING)

sb_idx = s.index_cast(pto.get_subblock_idx())
row_off_sb = sb_idx * c_s0_half

tv_o = as_tensor(o_tensor_ty, ptr=gm_o, shape=[s0, c_head], strides=[c_head, c1])

if KV_SPLIT > 1:
    partial_views = partial_tensor_views(gm_slot_buffer, c1, c_head)
    tv_o_parts = partial_views["o"]
    tv_m_parts = partial_views["m"]
    tv_l_parts = partial_views["l"]

qk_entry = pto.declare_global(qk_vec_slot_ty)
p_entry = pto.declare_global(p_slot_ty)
pv_entry = pto.declare_global(pv_vec_slot_ty)

# ---- emit_softmax(exp_max_slot, is_init): one streaming softmax ------
# Translates pto_macro_fa_softmax: row_max on unscaled QK -> row diff
# -> scale -> stream-update (running_max, running_sum) -> exp -> cvt
# -> push P. Keeping running_max unscaled matches the manual macro.
def emit_softmax(exp_max_slots, is_init):
    # Pop the wide QK slot (full subblock) and talloc one wide P slot;
    # iterate TILE_FACTOR row_slices, doing per-slice softmax math on
    # [Vec_S0, S1_TILE] tiles and per-slice reduce state. After all
    # row_slices, push the wide P slot.
    pto.tpop_into(qk_entry, qk_pipe, SPLIT_UP_DOWN)
    pto.talloc(p_entry, p_pipe, SPLIT_UP_DOWN)
    for row_slice in range(TILE_FACTOR):
        slot_part = slice_view(
            qk_vec_slot_part_ty,
            source=qk_entry,
            offsets=[const(row_slice * Vec_S0), c0],
            sizes=[c_vec_s0, c_s1_tile],
        )
        pto.load(slot_part, qk_vec)
        qk = qk_vec
        lmax = local_max[row_slice]
        lsum = local_sum[row_slice]
        rmax = running_max[row_slice]
        rsum = running_sum[row_slice]
        exp_slot = exp_max_slots[row_slice]
        tile.row_max(qk, tmp, lmax)

        # Reshape reductions to row-major so scalar broadcast helpers work.
        local_max_r = tile.reshape(red_row_ty, lmax)
        running_max_r = tile.reshape(red_row_ty, rmax)
        running_sum_r = tile.reshape(red_row_ty, rsum)
        local_sum_r = tile.reshape(red_row_ty, lsum)
        exp_max_r = tile.reshape(red_row_ty, exp_slot)

        if is_init:
            tile.row_expand_sub(qk, lmax, p_fp32)
            tile.mov(local_max_r, running_max_r)
            tile.muls(p_fp32, scale, p_fp32)
            tile.exp(p_fp32, p_fp32)
            tile.row_sum(p_fp32, tmp, rsum)
        else:
            tile.max(local_max_r, running_max_r, local_max_r)
            tile.sub(running_max_r, local_max_r, exp_max_r)
            tile.mov(local_max_r, running_max_r)
            tile.row_expand_sub(qk, lmax, p_fp32)
            tile.muls(exp_max_r, scale, exp_max_r)
            tile.muls(p_fp32, scale, p_fp32)
            tile.exp(exp_max_r, exp_max_r)
            tile.exp(p_fp32, p_fp32)
            tile.mul(running_sum_r, exp_max_r, running_sum_r)
            tile.row_sum(p_fp32, tmp, lsum)
            tile.add(running_sum_r, local_sum_r, running_sum_r)

        tile.cvt(p_fp32, p_fp16)
        p_part = slice_view(
            p_slot_part_ty, source=p_entry, offsets=[const(row_slice * Vec_S0), c0], sizes=[c_vec_s0, c_s1_tile]
        )
        pto.store(p_fp16, p_part)
    pto.tpush(p_entry, p_pipe, SPLIT_UP_DOWN)
    pto.tfree(qk_pipe, SPLIT_UP_DOWN, entry=qk_entry)

# ---- emit_gu(exp_max_slots, is_init): rescale + add running O ------
# GU also runs per-row_slice: each row_slice owns its own o_tile and
# pv_vec, indexed by the same exp_max_slots used during softmax.
def emit_gu(exp_max_slots, is_init):
    pto.tpop_into(pv_entry, pv_pipe, SPLIT_UP_DOWN)
    for row_slice in range(TILE_FACTOR):
        pv_part = slice_view(
            pv_vec_slot_part_ty,
            source=pv_entry,
            offsets=[const(row_slice * Vec_S0), c0],
            sizes=[c_vec_s0, c_head],
        )
        pto.load(pv_part, pv_vec[row_slice])
        if is_init:
            tile.mov(pv_vec[row_slice], o_tile[row_slice])
        else:
            tile.row_expand_mul(o_tile[row_slice], exp_max_slots[row_slice], o_tile[row_slice])
            tile.add(o_tile[row_slice], pv_vec[row_slice], o_tile[row_slice])
    pto.tfree(pv_pipe, SPLIT_UP_DOWN, entry=pv_entry)

def emit_ring_dispatch(tile_id, emit_fn):
    mod = tile_id % c_exp_ring
    emit_ring_branch(mod, emit_fn, 0)

def emit_ring_branch(mod, emit_fn, slot_idx):
    if slot_idx == EXP_RING - 1:
        emit_fn(exp_max_ring[slot_idx], is_init=False)
        return
    with pto.if_context(mod == const(slot_idx), has_else=True) as branch:
        emit_fn(exp_max_ring[slot_idx], is_init=False)
    with branch.else_context():
        emit_ring_branch(mod, emit_fn, slot_idx + 1)

def emit_softmax_dispatch(tile_id):
    emit_ring_dispatch(tile_id, emit_softmax)

def emit_gu_update_dispatch(tile_id):
    emit_ring_dispatch(tile_id, emit_gu)

def emit_gu_any(tile_id):
    with pto.if_context(tile_id == c0, has_else=True) as branch:
        emit_gu(exp_max_ring[0], is_init=True)
    with branch.else_context():
        emit_gu_update_dispatch(tile_id)

def vec_unit_body(steady_count):
    # Streaming softmax over this unit's KV tiles; identical structure
    # to the baseline per-qb body, parameterized by the tile count.
    # ---- vec prologue: softmax(0..QK_PRELOAD-1) --------------------
    for kp in range(QK_PRELOAD):
        emit_softmax(exp_max_ring[kp], is_init=(kp == 0))

    # ---- vec steady state. Match the 140tflops order: drain the
    # current PV/GU tile before producing the future P tile.
    with pto.if_context(steady_count > c0):
        emit_gu(exp_max_ring[0], is_init=True)
        emit_softmax(exp_max_ring[QK_PRELOAD % EXP_RING], is_init=False)

        for tile_id in pto.range(c1, steady_count, c1):
            next_tile = tile_id + c_preload
            emit_gu_update_dispatch(tile_id)
            emit_softmax_dispatch(next_tile)

    # ---- vec epilogue: drain last QK_PRELOAD gus -------------------
    for k in range(QK_PRELOAD):
        emit_gu_any(steady_count + const(k))

if KV_SPLIT == 1:
    for qb in pto.range(qb_start, qb_end, c1):
        vec_unit_body(steady_tiles)
        # Final divide + GM store, one row_slice at a time.
        for row_slice in range(TILE_FACTOR):
            tile.row_expand_div(o_tile[row_slice], running_sum[row_slice], o_tile[row_slice])
            o_view = slice_view(
                o_sub_slice_ty,
                source=tv_o,
                offsets=[qb * c_s0 + row_off_sb + const(row_slice * Vec_S0), c0],
                sizes=[c_vec_s0, c_head],
            )
            pto.store(o_tile[row_slice], o_view)
else:
    # ---- T1-A split-KV: stream this unit's KV chunk, then write the
    # UNNORMALIZED partial (O_acc, running_max m, running_sum l) keyed by
    # unit u. reduce_kernel flash-combines the KV_SPLIT partials per qb.
    c_kv_split = const(KV_SPLIT)
    tiles_per_chunk = num_tiles_s1 // c_kv_split
    steady_chunk = tiles_per_chunk - c_preload
    for u in pto.range(qb_start, qb_end, c1):
        vec_unit_body(steady_chunk)
        u_row_base = u * c_s0
        for row_slice in range(TILE_FACTOR):
            row = u_row_base + row_off_sb + const(row_slice * Vec_S0)
            o_part = slice_view(o_sub_slice_ty, source=tv_o_parts, offsets=[row, c0], sizes=[c_vec_s0, c_head])
            pto.store(o_tile[row_slice], o_part)
            m_part = slice_view(red_part_sub_ty, source=tv_m_parts, offsets=[row, c0], sizes=[c_vec_s0, c1])
            pto.store(running_max[row_slice], m_part)
            l_part = slice_view(red_part_sub_ty, source=tv_l_parts, offsets=[row, c0], sizes=[c_vec_s0, c1])
            pto.store(running_sum[row_slice], l_part)
