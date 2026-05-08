#!/usr/bin/env python3
"""
Compute UB and L1 buffer usage for FlashAttention kernel.

Based on allocate_vec_tile_buffers and allocate_cube_tile_buffers in fa_performance_dn_kernel.cpp.

Usage Examples:
    # Separate allocation (qkVecTile and pvVecTile use separate UB)
    python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 128 --cube_s1 64 --head_size 128 --tile_s1 128

    # Union allocation (qkVecTile and pvVecTile share UB, saves space)
    python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 128 --cube_s1 64 --head_size 128 --tile_s1 128 --union

    # With custom CV FIFO size
    python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 128 --cube_s1 64 --head_size 128 --tile_s1 128 --cv_fifo_size 8

    # Large parameters (shows overflow detection)
    python3 scripts/compute_buffer_usage_fa3_fp16_dn.py --cube_s0 256 --cube_s1 128 --head_size 256 --tile_s1 256
"""

import argparse


def compute_l1_usage(cube_s0: int, cube_s1: int, head_size: int) -> dict:
    """
    Compute L1 buffer usage for cube tiles in FlashAttention kernel.
    
    Args:
        cube_s0: Per-block rows for cube matmul
        cube_s1: Per-tile S1 chunk size
        head_size: Inner dimension (HEAD_SIZE)
    
    Returns:
        Dictionary with L1 usage details
    """
    # Buffer counts from kernel
    q_mat_tn_buffers = 1
    k_mat_tn_buffers = 2
    p_mat_tn_buffers = 2
    v_mat_tn_buffers = 2
    
    # Tile shapes: (Rows, Cols, DType)
    # TileMatQData: Tile<Mat, half, HEAD_SIZE, Cube_S0>
    tile_mat_q_shape = (head_size, cube_s0, "half")
    tile_mat_q_bytes = head_size * cube_s0 * 2
    
    # TileMatKData: Tile<Mat, half, Cube_S1, HEAD_SIZE>
    tile_mat_k_shape = (cube_s1, head_size, "half")
    tile_mat_k_bytes = cube_s1 * head_size * 2
    
    # TileMatPData: Tile<Mat, half, Cube_S0, Cube_S1>
    tile_mat_p_shape = (cube_s0, cube_s1, "half")
    tile_mat_p_bytes = cube_s0 * cube_s1 * 2
    
    # TileMatVData: Tile<Mat, half, Cube_S1, HEAD_SIZE>
    tile_mat_v_shape = (cube_s1, head_size, "half")
    tile_mat_v_bytes = cube_s1 * head_size * 2
    
    q_bytes = tile_mat_q_bytes * q_mat_tn_buffers
    k_bytes = tile_mat_k_bytes * k_mat_tn_buffers
    p_bytes = tile_mat_p_bytes * p_mat_tn_buffers
    v_bytes = tile_mat_v_bytes * v_mat_tn_buffers
    
    total_bytes = q_bytes + k_bytes + p_bytes + v_bytes
    
    MAX_TILE_L1_BYTES = 512 * 1024  # 512KB
    
    result = {
        "tile_shapes": {
            "TileMatQData": tile_mat_q_shape,
            "TileMatKData": tile_mat_k_shape,
            "TileMatPData": tile_mat_p_shape,
            "TileMatVData": tile_mat_v_shape,
        },
        "tile_sizes": {
            "TileMatQData": tile_mat_q_bytes,
            "TileMatKData": tile_mat_k_bytes,
            "TileMatPData": tile_mat_p_bytes,
            "TileMatVData": tile_mat_v_bytes,
        },
        "buffer_counts": {
            "qMatTNBuffers": q_mat_tn_buffers,
            "kMatTNBuffers": k_mat_tn_buffers,
            "pMatTNBuffers": p_mat_tn_buffers,
            "vMatTNBuffers": v_mat_tn_buffers,
        },
        "breakdown": {
            f"qMatTile[{q_mat_tn_buffers}]": q_bytes,
            f"kMatTile[{k_mat_tn_buffers}]": k_bytes,
            f"pMatTile[{p_mat_tn_buffers}]": p_bytes,
            f"vMatTile[{v_mat_tn_buffers}]": v_bytes,
        },
        "total_bytes": total_bytes,
        "max_l1_bytes": MAX_TILE_L1_BYTES,
        "utilization_pct": (total_bytes / MAX_TILE_L1_BYTES) * 100,
        "fits_in_l1": total_bytes <= MAX_TILE_L1_BYTES,
    }
    
    return result


def compute_buffer_usage_fa3_fp16_dn(cube_s0: int, cube_s1: int, head_size: int, tile_s1: int,
                     cv_fifo_size: int, use_union: bool) -> dict:
    """
    Compute UB usage for vector tiles in FlashAttention kernel.
    
    Args:
        cube_s0: Per-block rows for cube matmul
        cube_s1: Per-tile S1 chunk size
        head_size: Inner dimension (HEAD_SIZE)
        tile_s1: Logical tile size along S1
        cv_fifo_size: CV FIFO size (controls l1_exp_max_ififo array size)
        use_union: True for union allocation (qkVecTile/pvVecTile share UB),
                   False for separate allocation
    
    Returns:
        Dictionary with UB usage details
    """
    VEC_CORES = 2
    src_vec_tn_buffers = 2
    xexp_vec_tn_buffers = 2
    out_o_tile_n_buffers = 2
    
    k_tile_factor = tile_s1 // cube_s1
    
    vec_s0 = cube_s0 // VEC_CORES // k_tile_factor
    vec_gu_rows = cube_s0 // VEC_CORES
    subblock_rows = cube_s0 // VEC_CORES
    
    # Tile shapes: (Rows, Cols, DType)
    # TileDataF_T: Tile<Vec, float, Tile_S1, Vec_S0>
    tile_data_f_shape = (tile_s1, vec_s0, "float")
    tile_data_f_bytes = tile_s1 * vec_s0 * 4
    
    # ReduceTileF_T: Tile<Vec, float, 1, SubblockRows>
    reduce_tile_f_shape = (1, subblock_rows, "float")
    reduce_tile_f_bytes = subblock_rows * 4
    
    # TileDataH_T: Tile<Vec, half, Tile_S1, Vec_S0>
    tile_data_h_shape = (tile_s1, vec_s0, "half")
    tile_data_h_bytes = tile_s1 * vec_s0 * 2
    
    # TileOutGuT: Tile<Vec, float, VecGuRows, HEAD_SIZE>
    tile_out_gu_shape = (vec_gu_rows, head_size, "float")
    tile_out_gu_bytes = vec_gu_rows * head_size * 4
    
    # TileDataH_NZ_T: Tile<Vec, half, NzBufRows, Vec_S0> where NzBufRows = Cube_S1 + 1
    nz_buf_rows = cube_s1 + 1
    tile_data_h_nz_shape = (nz_buf_rows, vec_s0, "half")
    tile_data_h_nz_bytes = nz_buf_rows * vec_s0 * 2
    
    src_bytes = tile_data_f_bytes * src_vec_tn_buffers
    pv_bytes = tile_out_gu_bytes * out_o_tile_n_buffers
    xexp_bytes = tile_data_h_bytes * xexp_vec_tn_buffers
    
    exp_max_buffers = cv_fifo_size
    
    # Note: input_reduce_tmp removed (can reuse qkVecTile buffer)
    
    if use_union:
        union_stride = max(tile_data_f_bytes, tile_out_gu_bytes)
        union_bytes = union_stride * src_vec_tn_buffers
        total_bytes = union_bytes + xexp_bytes + \
                      (reduce_tile_f_bytes * (3 + exp_max_buffers)) + tile_out_gu_bytes
    else:
        total_bytes = src_bytes + pv_bytes + xexp_bytes + \
                      (reduce_tile_f_bytes * (3 + exp_max_buffers)) + tile_out_gu_bytes
        union_stride = 0
    
    # nzConvBuffer is allocated at the end of UB (separate from main allocation)
    total_bytes += tile_data_h_nz_bytes
    
    MAX_VEC_UB_BYTES = 256 * 1024
    
    result = {
        "parameters": {
            "CUBE_S0": cube_s0,
            "CUBE_S1": cube_s1,
            "HEAD_SIZE": head_size,
            "TILE_S1": tile_s1,
            "CV_FIFO_SIZE": cv_fifo_size,
            "use_union": use_union,
        },
        "derived": {
            "VEC_CORES": VEC_CORES,
            "kTileFactor": k_tile_factor,
            "Vec_S0": vec_s0,
            "VecGuRows": vec_gu_rows,
            "SubblockRows": subblock_rows,
            "NzBufRows": nz_buf_rows,
        },
        "tile_shapes": {
            "TileDataF_T": tile_data_f_shape,
            "ReduceTileF_T": reduce_tile_f_shape,
            "TileDataH_T": tile_data_h_shape,
            "TileOutGuT": tile_out_gu_shape,
            "TileDataH_NZ_T": tile_data_h_nz_shape,
        },
        "tile_sizes": {
            "TileDataF_T": tile_data_f_bytes,
            "ReduceTileF_T": reduce_tile_f_bytes,
            "TileDataH_T": tile_data_h_bytes,
            "TileOutGuT": tile_out_gu_bytes,
            "TileDataH_NZ_T": tile_data_h_nz_bytes,
        },
        "buffer_counts": {
            "srcVecTNBuffers": src_vec_tn_buffers,
            "xexpVecTNBuffers": xexp_vec_tn_buffers,
            "outOTileNBuffers": out_o_tile_n_buffers,
            "l1_exp_max_ififo": exp_max_buffers,
        },
        "total_bytes": total_bytes,
        "max_ub_bytes": MAX_VEC_UB_BYTES,
        "utilization_pct": (total_bytes / MAX_VEC_UB_BYTES) * 100,
        "fits_in_ub": total_bytes <= MAX_VEC_UB_BYTES,
    }
    
    # Add union info if applicable
    if use_union:
        result["union_info"] = {
            "stride": union_stride,
            "buffers": src_vec_tn_buffers,
        }
    
    if use_union:
        result["breakdown"] = {
            "runningOTile": tile_out_gu_bytes,
            f"union(qkVecTile/pvVecTile)[{src_vec_tn_buffers}]": union_stride * src_vec_tn_buffers,
            "m1_local_max": reduce_tile_f_bytes,
            "m2_global_max": reduce_tile_f_bytes,
            "l1_local_sum": reduce_tile_f_bytes,
            "l2_global_sum": reduce_tile_f_bytes,
            f"l1_exp_max[{exp_max_buffers}]": reduce_tile_f_bytes * exp_max_buffers,
            f"x_expT[{xexp_vec_tn_buffers}]": xexp_bytes,
            "nzConvBuffer": tile_data_h_nz_bytes,
        }
    else:
        result["breakdown"] = {
            f"qkVecTile[{src_vec_tn_buffers}]": src_bytes,
            "runningOTile": tile_out_gu_bytes,
            f"pvVecTile[{out_o_tile_n_buffers}]": pv_bytes,
            "m1_local_max": reduce_tile_f_bytes,
            "m2_global_max": reduce_tile_f_bytes,
            "l1_local_sum": reduce_tile_f_bytes,
            "l2_global_sum": reduce_tile_f_bytes,
            f"l1_exp_max[{exp_max_buffers}]": reduce_tile_f_bytes * exp_max_buffers,
            f"x_expT[{xexp_vec_tn_buffers}]": xexp_bytes,
            "nzConvBuffer": tile_data_h_nz_bytes,
        }
    
    return result


def format_size(bytes_val: int) -> str:
    """Format bytes as human-readable string."""
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB ({bytes_val} bytes)"
    return f"{bytes_val} bytes"


def format_size_short(bytes_val: int) -> str:
    """Format bytes as short human-readable string for table."""
    if bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


def print_allocation_table(title: str, breakdown: dict, tile_shapes: dict, tile_sizes: dict,
                           total_bytes: int, max_bytes: int, utilization_pct: float,
                           fits_in_buffer: bool, var_to_tile_type: dict, buffer_name: str,
                           use_union: bool = False, union_stride: int = 0, union_bufs: int = 0):
    """Print allocation table for UB or L1 buffer."""
    print(f"\n[{title}]")
    print("-" * 120)
    
    print(f"{'Variable Name':<35} {'Tile Type':<18} {'Shape (R x C)':<14} {'DType':<8} {'#Bufs':<6} {'Size/Tile':<12} {'Total Size':<12}")
    print("-" * 120)
    
    for var_name, size in breakdown.items():
        base_name = var_name.split("[")[0] if "[" in var_name else var_name
        is_union_var = "union(qkVecTile/pvVecTile)" in var_name
        
        tile_type = var_to_tile_type.get(base_name if not is_union_var else "union", base_name)
        
        if is_union_var:
            # Show union info with actual tile shapes
            tile_type_str = "TileDataF_T/TileOutGuT"
            qk_shape = tile_shapes.get("TileDataF_T", ("-", "-", "-"))
            pv_shape = tile_shapes.get("TileOutGuT", ("-", "-", "-"))
            qk_rows, qk_cols, qk_dtype = qk_shape
            pv_rows, pv_cols, pv_dtype = pv_shape
            shape_str = f"{qk_rows}x{qk_cols}/{pv_rows}x{pv_cols}"
            dtype_str = f"{qk_dtype}/{pv_dtype}"
            num_bufs = str(union_bufs)
            size_per_tile = format_size_short(union_stride)
        elif tile_type in tile_shapes:
            rows, cols, dtype = tile_shapes[tile_type]
            shape_str = f"{rows} x {cols}"
            dtype_str = dtype
            tile_type_str = tile_type
            
            if "[" in var_name:
                buf_count_str = var_name.split("[")[1].split("]")[0]
                num_bufs = buf_count_str
                size_per_tile = format_size_short(tile_sizes[tile_type])
            else:
                num_bufs = "1"
                size_per_tile = format_size_short(size)
        else:
            tile_type_str = "-"
            shape_str = "-"
            dtype_str = "-"
            num_bufs = "-"
            size_per_tile = "-"
        
        total_str = format_size_short(size)
        print(f"{var_name:<35} {tile_type_str:<18} {shape_str:<14} {dtype_str:<8} {num_bufs:<6} {size_per_tile:<12} {total_str:<12}")
    
    print("-" * 120)
    print(f"{'TOTAL':<35} {'':<18} {'':<14} {'':<8} {'':<6} {'':<12} {format_size_short(total_bytes):<12}")
    print("-" * 120)
    
    print(f"\n[{buffer_name} Summary]")
    print(f"  Max {buffer_name} capacity: {format_size(max_bytes)}")
    print(f"  Utilization: {utilization_pct:.1f}%")
    
    if fits_in_buffer:
        print(f"  Status: PASS (fits in {max_bytes // 1024}KB {buffer_name})")
    else:
        overflow = total_bytes - max_bytes
        print(f"  Status: FAIL (overflow by {format_size(overflow)})")


def main():
    parser = argparse.ArgumentParser(
        description="Compute UB and L1 buffer usage for FlashAttention kernel"
    )
    parser.add_argument("--cube_s0", type=int, required=True,
                        help="Per-block rows for cube matmul (CUBE_S0)")
    parser.add_argument("--cube_s1", type=int, required=True,
                        help="Per-tile S1 chunk size (CUBE_S1)")
    parser.add_argument("--head_size", type=int, required=True,
                        help="Inner dimension (HEAD_SIZE)")
    parser.add_argument("--tile_s1", type=int, required=True,
                        help="Logical tile size along S1 (TILE_S1)")
    parser.add_argument("--cv_fifo_size", type=int, default=4,
                        help="CV FIFO size (default: 4)")
    parser.add_argument("--union", action="store_true",
                        help="Use union allocation for qkVecTile/pvVecTile (saves UB)")
    
    args = parser.parse_args()
    
    if args.tile_s1 % args.cube_s1 != 0:
        print(f"Error: TILE_S1 ({args.tile_s1}) must be divisible by CUBE_S1 ({args.cube_s1})")
        return
    
    # Compute L1 usage
    l1_result = compute_l1_usage(
        cube_s0=args.cube_s0,
        cube_s1=args.cube_s1,
        head_size=args.head_size,
    )
    
    # Compute UB usage
    ub_result = compute_buffer_usage_fa3_fp16_dn(
        cube_s0=args.cube_s0,
        cube_s1=args.cube_s1,
        head_size=args.head_size,
        tile_s1=args.tile_s1,
        cv_fifo_size=args.cv_fifo_size,
        use_union=args.union,
    )
    
    print("=" * 95)
    print("Buffer Usage Analysis for FlashAttention Kernel")
    print("=" * 95)
    
    print("\n[Parameters]")
    for k, v in ub_result["parameters"].items():
        print(f"  {k}: {v}")
    
    print("\n[Derived Values]")
    for k, v in ub_result["derived"].items():
        print(f"  {k}: {v}")
    
    # L1 allocation table
    l1_var_to_tile_type = {
        "qMatTile": "TileMatQData",
        "kMatTile": "TileMatKData",
        "pMatTile": "TileMatPData",
        "vMatTile": "TileMatVData",
    }
    
    print_allocation_table(
        title="L1 Allocation Table",
        breakdown=l1_result["breakdown"],
        tile_shapes=l1_result["tile_shapes"],
        tile_sizes=l1_result["tile_sizes"],
        total_bytes=l1_result["total_bytes"],
        max_bytes=l1_result["max_l1_bytes"],
        utilization_pct=l1_result["utilization_pct"],
        fits_in_buffer=l1_result["fits_in_l1"],
        var_to_tile_type=l1_var_to_tile_type,
        buffer_name="L1",
    )
    
    # UB allocation table
    ub_var_to_tile_type = {
        "qkVecTile": "TileDataF_T",
        "pvVecTile": "TileOutGuT",
        "union(qkVecTile/pvVecTile)": "union",
        "runningOTile": "TileOutGuT",
        "x_expT": "TileDataH_T",
        "m1_local_max": "ReduceTileF_T",
        "m2_global_max": "ReduceTileF_T",
        "l1_local_sum": "ReduceTileF_T",
        "l2_global_sum": "ReduceTileF_T",
        "l1_exp_max": "ReduceTileF_T",
        "nzConvBuffer": "TileDataH_NZ_T",
    }
    
    # Get union info if applicable
    union_stride = 0
    union_bufs = 0
    if args.union and "union_info" in ub_result:
        union_stride = ub_result["union_info"]["stride"]
        union_bufs = ub_result["union_info"]["buffers"]
    
    print_allocation_table(
        title="UB Allocation Table",
        breakdown=ub_result["breakdown"],
        tile_shapes=ub_result["tile_shapes"],
        tile_sizes=ub_result["tile_sizes"],
        total_bytes=ub_result["total_bytes"],
        max_bytes=ub_result["max_ub_bytes"],
        utilization_pct=ub_result["utilization_pct"],
        fits_in_buffer=ub_result["fits_in_ub"],
        var_to_tile_type=ub_var_to_tile_type,
        buffer_name="UB",
        use_union=args.union,
        union_stride=union_stride,
        union_bufs=union_bufs,
    )


if __name__ == "__main__":
    main()
