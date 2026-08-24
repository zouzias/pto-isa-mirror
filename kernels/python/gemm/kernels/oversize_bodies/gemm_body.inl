/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

with pto.cube_section():
    c0 = const(0)
    c1 = const(1)
    c2 = const(2)
    c_m = const(m)
    c_k = const(k)
    c_n = const(n)
    c_base_m = const(base_m)
    c_base_k = const(base_k)
    c_base_n = const(base_n)
    c_step_ka = const(step_ka)
    c_swizzle_count_n = const(swizzle_count_n)
    c_swizzle_count_n_m1 = c_swizzle_count_n - c1
    c_panel_k = c_base_k * c_step_ka
    bid = s.index_cast(pto.get_block_idx())
    block_num = s.index_cast(pto.get_block_num())

    m_tiles = c_m // c_base_m
    n_tiles = c_n // c_base_n
    output_tiles = m_tiles * n_tiles
    k_iters = c_k // c_base_k
    k_panel_iters = k_iters // c_step_ka

    # GM tensor views. B uses DN layout because the host passes
    # contiguous transposed B storage while the logical GEMM still sees
    # B as [K, N].
    tv_a = pto.as_tensor(
        tensor_type_in,
        ptr=a_ptr,
        shape=[c_m, c_k],
        strides=[c_k, c1],
    )
    tv_b = pto.as_tensor(
        tensor_type_in,
        ptr=b_ptr,
        shape=[c_k, c_n],
        strides=[c1, c_k],
        layout="DN",
    )
    tv_c = pto.as_tensor(
        tensor_type_out,
        ptr=out_ptr,
        shape=[c_m, c_n],
        strides=[c_n, c1],
    )

    # L1 panels are double buffered; L0A/L0B ping-pong buffers feed the
    # unrolled K steps inside one panel.
    a_l1 = [pto.alloc_tile(tile_buf_a_mat), pto.alloc_tile(tile_buf_a_mat)]
    b_l1 = [pto.alloc_tile(tile_buf_b_mat), pto.alloc_tile(tile_buf_b_mat)]
    a_l0 = [pto.alloc_tile(tile_buf_a_tile), pto.alloc_tile(tile_buf_a_tile)]
    b_l0 = [pto.alloc_tile(tile_buf_b_tile), pto.alloc_tile(tile_buf_b_tile)]
    c_l0 = pto.alloc_tile(tile_buf_c_tile)

    def load_panel(l1_idx, panel_idx, m_offset, n_offset):
        # Load one A/B K panel from GM into the selected L1 slots.
        k_panel_offset = panel_idx * c_panel_k
        sv_a = pto.slice_view(
            tile_view_a_mat,
            source=tv_a,
            offsets=[m_offset, k_panel_offset],
            sizes=[c_base_m, c_panel_k],
        )
        sv_b = pto.slice_view(
            tile_view_b_mat,
            source=tv_b,
            offsets=[k_panel_offset, n_offset],
            sizes=[c_panel_k, c_base_n],
        )
        pto.load(sv_a, a_l1[l1_idx])
        pto.load(sv_b, b_l1[l1_idx])

    def run_unrolled_k(l1_idx, panel_idx, first_panel):
        # Move one L1 panel through L0 and accumulate into c_l0.
        for inner in range(step_ka):
            l0_idx = inner % 2
            k_inner_offset = const(base_k * inner)
            tile.extract(a_l1[l1_idx], c0, k_inner_offset, a_l0[l0_idx])
            tile.extract(b_l1[l1_idx], k_inner_offset, c0, b_l0[l0_idx])
            if first_panel and inner == 0:
                # The first K slice initializes ACC; all remaining
                # slices accumulate into the same C tile.
                pto.cond(
                    s.eq(panel_idx, c0),
                    lambda: tile.matmul(a_l0[l0_idx], b_l0[l0_idx], c_l0),
                    lambda: tile.matmul_acc(c_l0, a_l0[l0_idx], b_l0[l0_idx], c_l0),
                )
            else:
                tile.matmul_acc(c_l0, a_l0[l0_idx], b_l0[l0_idx], c_l0)

    for tile_id in pto.range(bid, output_tiles, block_num):
        # N-group swizzle over the base-tile grid. swizzle_count_n=1
        # degenerates to row-major order over the base-tile grid.
        tile_block_loop = (n_tiles + c_swizzle_count_n_m1) // c_swizzle_count_n
        tile_block_span = c_swizzle_count_n * m_tiles
        tile_block_idx = tile_id // tile_block_span
        in_tile_block_idx = tile_id % tile_block_span
        is_last_block = tile_block_idx == (tile_block_loop - c1)
        n_col_tail = n_tiles - c_swizzle_count_n * tile_block_idx
        n_col = s.select(is_last_block, n_col_tail, c_swizzle_count_n)
        base_m_idx = in_tile_block_idx // n_col
        base_n_idx = tile_block_idx * c_swizzle_count_n + (in_tile_block_idx % n_col)
        odd_block = (tile_block_idx % c2) == c1
        flipped_m_idx = m_tiles - base_m_idx - c1
        base_m_idx = s.select(odd_block, flipped_m_idx, base_m_idx)

        m_offset = base_m_idx * c_base_m
        n_offset = base_n_idx * c_base_n

        for panel_idx in pto.range(c0, k_panel_iters, c2):
            next_panel_idx = panel_idx + c1
            load_panel(0, panel_idx, m_offset, n_offset)
            load_panel(1, next_panel_idx, m_offset, n_offset)
            run_unrolled_k(0, panel_idx, first_panel=True)
            run_unrolled_k(1, next_panel_idx, first_panel=False)

        sv_out = pto.slice_view(
            tile_view_out,
            source=tv_c,
            offsets=[m_offset, n_offset],
            sizes=[c_base_m, c_base_n],
        )
        pto.store(c_l0, sv_out)
