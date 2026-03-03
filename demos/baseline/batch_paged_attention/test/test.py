#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import op_extension
import numpy as np

# Configuration matching CPU demo and kernel constants
BATCH = 2
NUM_HEADS = 16
KV_HEAD_NUM = 1
HEAD_DIM = 16
BLOCK_SIZE = 16
MAX_NUM_BLOCKS = 4
TOTAL_BLOCKS = 8
CONTEXT_LENS = [33, 17]
SCALE = 1.0
RNG_SEED = 20251220


def batch_paged_attention_reference(query, key_cache, value_cache, block_table, context_lens):
    """
    Reference implementation matching the CPU demo and simpler/examples golden.py.
    Uses online softmax with fp16 truncation for pij.

    Args:
        query:       np.float16 (batch * num_heads, head_dim)
        key_cache:   np.float16 (total_blocks * block_size, head_dim)
        value_cache: np.float16 (total_blocks * block_size, head_dim)
        block_table: np.int32   (batch, max_num_blocks)
        context_lens: np.int32  (batch,)

    Returns:
        output: np.float32 (batch * num_heads, head_dim)
    """
    batch = len(context_lens)
    output = np.zeros((batch * NUM_HEADS, HEAD_DIM), dtype=np.float32)

    # Compute global max_bn across all batches (matching simpler orchestration)
    global_max_bn = 0
    for b in range(batch):
        bn_b = (context_lens[b] + BLOCK_SIZE - 1) // BLOCK_SIZE
        global_max_bn = max(global_max_bn, bn_b)

    for b in range(batch):
        ctx_len = context_lens[b]

        for h in range(NUM_HEADS):
            oi = np.zeros(HEAD_DIM, dtype=np.float32)
            mi = -np.inf
            li = 0.0

            for bn in range(global_max_bn):
                start = bn * BLOCK_SIZE
                valid_len = min(BLOCK_SIZE, max(0, ctx_len - start))

                if valid_len == 0:
                    # Block beyond sequence: mij=-1e30, lij=0, oi_new=0
                    mij_val = -1e30
                    mi_new = max(mi, mij_val)
                    alpha = np.exp(mi - mi_new)
                    li = alpha * li
                    oi = alpha * oi
                    mi = mi_new
                    continue

                phys_block = block_table[b, bn]

                # sij = q @ k^T (float32 computation)
                sij = np.zeros(BLOCK_SIZE, dtype=np.float32)
                for j in range(BLOCK_SIZE):
                    dot = 0.0
                    for d in range(HEAD_DIM):
                        dot += float(query[(b * NUM_HEADS + h) * HEAD_DIM + d]) * \
                               float(key_cache[(phys_block * BLOCK_SIZE + j) * HEAD_DIM + d])
                    sij[j] = dot

                # Mask invalid positions first (before scale)
                for j in range(valid_len, BLOCK_SIZE):
                    sij[j] = -np.inf

                # Apply scale
                sij *= SCALE

                # Row max (clamp to -1e30)
                mij = max(np.max(sij), -1e30)

                # Exp
                pij = np.exp(sij - mij).astype(np.float32)

                # FP16 truncation
                pij = pij.astype(np.float16).astype(np.float32)

                # Row sum
                lij = np.sum(pij)

                # P @ V
                oi_new = np.zeros(HEAD_DIM, dtype=np.float32)
                for d in range(HEAD_DIM):
                    for j in range(BLOCK_SIZE):
                        oi_new[d] += pij[j] * float(
                            value_cache[(phys_block * BLOCK_SIZE + j) * HEAD_DIM + d])

                # Online softmax update
                mi_new = max(mi, mij)
                alpha = np.exp(mi - mi_new)
                beta = np.exp(mij - mi_new)
                li = alpha * li + beta * lij
                oi = alpha * oi + beta * oi_new
                mi = mi_new

            # Final normalization
            inv_li = (1.0 / li) if li > 0.0 else 0.0
            output[b * NUM_HEADS + h] = oi * inv_li

    return output


def gen_test_data(batch=BATCH, context_lens=None, seed=RNG_SEED):
    """Generate test data matching CPU demo's random generation."""
    if context_lens is None:
        context_lens = CONTEXT_LENS[:batch]

    rng = np.random.RandomState(seed)

    q_total = batch * NUM_HEADS * HEAD_DIM
    kv_total = TOTAL_BLOCKS * BLOCK_SIZE * HEAD_DIM

    # Random block table
    block_table = rng.randint(0, TOTAL_BLOCKS, size=(batch, MAX_NUM_BLOCKS)).astype(np.int32)

    # Q/K with uniform(-0.5, 0.5), V with uniform(-1.0, 1.0), all in fp16
    query = rng.uniform(-0.5, 0.5, size=q_total).astype(np.float16)
    key_cache = rng.uniform(-0.5, 0.5, size=kv_total).astype(np.float16)
    value_cache = rng.uniform(-1.0, 1.0, size=kv_total).astype(np.float16)

    context_lens_arr = np.array(context_lens, dtype=np.int32)

    return query, key_cache, value_cache, block_table, context_lens_arr


class TestBatchPagedAttention(TestCase):

    def test_bpa_custom_ops(self):
        """Test batch paged attention against reference implementation."""
        query, key_cache, value_cache, block_table, context_lens = gen_test_data()

        # Compute reference output on CPU
        ref_output = batch_paged_attention_reference(query, key_cache, value_cache, block_table, context_lens)

        # Move tensors to NPU
        query_npu = torch.from_numpy(query).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        block_table_npu = torch.from_numpy(block_table).npu()
        context_lens_npu = torch.from_numpy(context_lens).npu()

        # Call the custom operator
        output = torch.ops.npu.my_bpa(query_npu, key_cache_npu, value_cache_npu,
                                      block_table_npu, context_lens_npu)

        # Move output to CPU for comparison
        output_cpu = output.cpu().numpy()
        ref_torch = torch.from_numpy(ref_output)
        out_torch = torch.from_numpy(output_cpu)

        # Print detailed diagnostic info before assertion
        prec = 1e-2
        abs_diff = np.abs(output_cpu - ref_output)
        max_abs_diff = np.max(abs_diff)
        max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        ref_abs = np.abs(ref_output)
        rel_diff = np.where(ref_abs > 0, abs_diff / ref_abs, 0.0)
        max_rel_diff = np.max(rel_diff)
        max_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)
        num_mismatch = int(np.sum(abs_diff > prec))

        print(f"\n{'=' * 70}")
        print(f"Diagnostic Info (B={BATCH}, H={NUM_HEADS}, D={HEAD_DIM}, "
              f"BS={BLOCK_SIZE}, context_lens={list(context_lens)})")
        print(f"{'=' * 70}")
        print(f"  Output shape : {output_cpu.shape}, dtype: {output_cpu.dtype}")
        print(f"  Ref    shape : {ref_output.shape}, dtype: {ref_output.dtype}")
        print(f"  Tolerance    : {prec}")
        print(f"  Max abs diff : {max_abs_diff:.6e} at index {max_abs_idx}")
        print(f"    -> output  : {output_cpu[max_abs_idx]:.6e}")
        print(f"    -> ref     : {ref_output[max_abs_idx]:.6e}")
        print(f"  Max rel diff : {max_rel_diff:.6e} at index {max_rel_idx}")
        print(f"    -> output  : {output_cpu[max_rel_idx]:.6e}")
        print(f"    -> ref     : {ref_output[max_rel_idx]:.6e}")
        print(f"  Mismatches   : {num_mismatch} / {abs_diff.size} "
              f"(>{prec} abs diff)")

        # Per-batch breakdown
        for b in range(BATCH):
            start = b * NUM_HEADS
            end = (b + 1) * NUM_HEADS
            batch_abs = abs_diff[start:end]
            batch_max = np.max(batch_abs)
            batch_mismatch = int(np.sum(batch_abs > prec))
            print(f"  Batch {b} (ctx_len={context_lens[b]}): "
                  f"max_abs_diff={batch_max:.6e}, mismatches={batch_mismatch}/{batch_abs.size}")

        # Show first few mismatched elements
        flat_mask = abs_diff.flatten() > prec
        mismatch_indices = np.where(flat_mask)[0]
        if len(mismatch_indices) > 0:
            show_n = min(10, len(mismatch_indices))
            print(f"\n  First {show_n} mismatched elements (flat index):")
            out_flat = output_cpu.flatten()
            ref_flat = ref_output.flatten()
            for i in range(show_n):
                idx = mismatch_indices[i]
                print(f"    [{idx}] output={out_flat[idx]:.6e}, "
                      f"ref={ref_flat[idx]:.6e}, diff={abs_diff.flatten()[idx]:.6e}")
        print(f"{'=' * 70}\n")

        # Validate results (tolerance matching CPU demo: abs=1e-2, rel=1e-2)
        self.assertTrue(
            max_abs_diff <= prec,
            f"Max absolute diff {max_abs_diff:.6e} exceeds tolerance {prec:.1e}. "
            f"Worst at index {max_abs_idx}: output={output_cpu[max_abs_idx]:.6e}, "
            f"ref={ref_output[max_abs_idx]:.6e}. "
            f"Total mismatches: {num_mismatch}/{abs_diff.size}"
        )

        print(f"[PASS] batch_paged_attention: B={BATCH} H={NUM_HEADS} D={HEAD_DIM} "
              f"BS={BLOCK_SIZE} context_lens={list(context_lens)}")


if __name__ == "__main__":
    run_tests()
