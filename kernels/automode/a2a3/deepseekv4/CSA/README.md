# CSA — Compressed Sparse FlashAttention

Auto-mode A3 split of the `sparse_attn` kernel from
[deepseek/kernel.py:277](../../../../../deepseek/kernel.py) into per-stage
leaves so each leaf is **either cube or vector** (never both).

## Reference

```python
# kernel.py:293-352, condensed
for t in T.Pipelined(num_blocks, num_stages=num_stages):
    # 1. gather KV by topk_idx  (vector — memory rearrangement)
    kv_shared[i,j] = kv[by, topk_idxs[by, bx, t*block + i], j]
    # 2. masked QK matmul       (cube)
    acc_s = q_shared @ kv_shared^T  * softmax_scale
    # 3. online softmax          (vector — running max, exp, rescale, sum)
    scores_max_prev = scores_max
    reduce_max(acc_s -> scores_max, dim=1)
    scores_scale = exp(scores_max_prev - scores_max)
    acc_s = exp(acc_s - scores_max)
    sum_exp = sum_exp * scores_scale + reduce_sum(acc_s, dim=1)
    # 4. acc_o rescale + PV matmul  (vector then cube)
    acc_o = acc_o * scores_scale
    acc_o += acc_s @ kv_shared
# 5. final output rescale + attn_sink  (vector)
sum_exp += exp(attn_sink - scores_max)
acc_o /= sum_exp
```

## Leaves

| Leaf | HW | Role | Source line |
|------|----|----- |-------------|
| [gather_kv/](gather_kv/) | vector | gather KV[topk_idx[t]] into shared tile | kernel.py:322-325 |
| [qk_matmul/](qk_matmul/) | cube | `acc_s = Q @ K^T * scale`, with -inf mask | kernel.py:326-330 |
| [online_softmax/](online_softmax/) | vector | running max / exp / rescale / sum | kernel.py:331-340 |
| [pv_matmul/](pv_matmul/) | cube | `acc_o += acc_s_cast @ V`, with prior rescale | kernel.py:341-343 |
| [output_rescale/](output_rescale/) | vector | final `/= sum_exp` + `attn_sink` log-sum-exp tail | kernel.py:345-350 |

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `H, D, S, BLOCK` where

- `H` = number of heads (padded to 16 if `h < 16`; see kernel.py:359)
- `D` = head dim
- `S` = `topk` length (per-query selected KV positions)
- `BLOCK` = inner pipeline block size along `S` (`block` in kernel.py:290)

`num_blocks = ceildiv(S, BLOCK)`. The default mirrors the TileLang setting:
`block=64`, `num_stages=2`.

## Auto-mode constraints (CSA family)

- A3 only (`PTO_NPU_ARCH_A2A3`).
- Each leaf is a single AICORE kernel; no `block_idx` work split.
- No `TASSIGN`, `TPipe`, `TPUSH`/`TPOP`, no raw `set_flag`/`wait_flag`.
- Cube leaves (`qk_matmul`, `pv_matmul`) reason in 16×16 fractals (L0A/L0B/L0C).
- Vector leaves (`gather_kv`, `online_softmax`, `output_rescale`) operate
  on UB tiles only.
