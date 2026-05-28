#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import json
from pathlib import Path


def build_routes(case_name, rank, rank_num, m, topk, expert_per_rank):
    expert_num = rank_num * expert_per_rank
    routes = []
    probs = []
    for token in range(m):
        for slot in range(topk):
            expert = (token * topk + slot + rank) % expert_num
            if case_name == "skewed":
                expert = rank % expert_num if slot == 0 else expert_num - 1
            elif case_name == "zero-token":
                expert = (token + slot) % max(1, expert_num - 1)
            routes.append(expert)
            probs.append(1.0 if topk == 1 else 1.0 / topk)
    return routes, probs


def token_per_expert_index(rank_num, expert_per_rank, token_owner_rank, expert_owner_rank, local_expert):
    return (token_owner_rank * rank_num + expert_owner_rank) * expert_per_rank + local_expert


def checksum(values):
    value_mask = (1 << 64) - 1
    hash_value = 1469598103934665603
    for value in values:
        hash_value ^= value & value_mask
        hash_value = (hash_value * 1099511628211) & value_mask
    return hash_value


def build_routing_reference(rank, rank_num, m, topk, expert_per_rank, expert_id):
    matrix = [0] * (rank_num * rank_num * expert_per_rank)
    expanded_row_idx = [-1] * (m * topk)
    dispatch_offset = [0] * (m * topk)
    local_cursor = [0] * (rank_num * expert_per_rank)
    expert_num = rank_num * expert_per_rank
    for token in range(m):
        for slot in range(topk):
            route = token * topk + slot
            expert = expert_id[route]
            if expert < 0 or expert >= expert_num:
                continue
            expert_owner_rank = expert // expert_per_rank
            local_expert = expert % expert_per_rank
            matrix_idx = token_per_expert_index(rank_num, expert_per_rank, rank, expert_owner_rank, local_expert)
            matrix[matrix_idx] += 1
            expanded_row_idx[route] = route
            cursor_idx = expert_owner_rank * expert_per_rank + local_expert
            dispatch_offset[route] = local_cursor[cursor_idx]
            local_cursor[cursor_idx] += 1
    return {
        "token_per_expert_matrix": matrix,
        "expanded_row_idx": expanded_row_idx,
        "dispatch_offset": dispatch_offset,
        "token_per_expert_matrix_checksum": checksum(matrix),
        "expanded_row_idx_checksum": checksum(expanded_row_idx),
        "dispatch_offset_checksum": checksum(dispatch_offset),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic M0 host reference data.")
    parser.add_argument("--case-name", choices=["small", "balanced", "skewed", "zero-token"], default="small")
    parser.add_argument("--rank-num", type=int, default=2)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--experts-per-rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--print-reference", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_values = [
        ((idx + args.seed + args.rank * 17) % 257) / 128.0 - 1.0
        for idx in range(args.tokens * args.hidden_size)
    ]
    expert_id, probs = build_routes(
        args.case_name,
        args.rank,
        args.rank_num,
        args.tokens,
        args.topk,
        args.experts_per_rank,
    )
    routing_reference = build_routing_reference(
        args.rank,
        args.rank_num,
        args.tokens,
        args.topk,
        args.experts_per_rank,
        expert_id,
    )
    payload = {
        "case_name": args.case_name,
        "rank_num": args.rank_num,
        "rank": args.rank,
        "tokens": args.tokens,
        "hidden_size": args.hidden_size,
        "topk": args.topk,
        "experts_per_rank": args.experts_per_rank,
        "seed": args.seed,
        "input": input_values,
        "expert_id": expert_id,
        "probs": probs,
        "routing_reference": routing_reference,
    }
    output = out_dir / f"rank{args.rank}_{args.case_name}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"generated {output}")
    if args.print_reference:
        print(f"token_per_expert_matrix={routing_reference['token_per_expert_matrix']}")
        print(f"expanded_row_idx={routing_reference['expanded_row_idx']}")


if __name__ == "__main__":
    main()
