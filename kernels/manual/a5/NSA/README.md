# NSA Flash Attention (A5 DN)

Native Sparse Attention (NSA) reference for Ascend A5, built on the [`../flash_atten`](../flash_atten) DN kernel.

Reference: [`Projects/attention/pytorch/NSA.py`](../../../../../../attention/pytorch/NSA.py) (DeepSeek NSA, arXiv 2502.11089).

---

## Algorithm (single query head)

NSA fuses three parallel flash-attention branches with learned gates:

```
o = g_cmp * FA(q, K_cmp, V_cmp) + g_slc * FA(q, K_slc, V_slc) + g_win * FA(q, K_win, V_win)
```

| Branch | Role | Branch S1 | Default (S1=512) |
|--------|------|-----------|---------------------|
| **Compression** | Block MLP compresses cmp K/V; coarse global attention | `S1 / BLOCK_SIZE` | 128 |
| **Selection** | Top-N blocks from compression importance scores | `NUM_SELECTED × SELECT_BLOCK_SIZE` | 128 |
| **Sliding window** | Recent tokens from win K/V | `WINDOW_SIZE` | 128 |

Each branch runs the standard MHA DN pipeline:

| Stage | Computation | Inner dim |
|-------|-------------|-----------|
| QK | `q [S0, head_size] @ k_branch^T` | `head_size` |
| Softmax | streaming softmax, scale `1/sqrt(head_size)` | — |
| PV | `P [S0, S1_branch] @ V [S1_branch, head_size]` | `head_size` |
| GU | running output rescale / divide | — |

Golden generation (`scripts/gen_data.py`) implements block compression MLP, aggregate top-N block selection, per-branch FA goldens, and sigmoid gates in Python. The A5 path runs **`LaunchTFA` three times** (one per branch) and fuses branch outputs with gate weights on the host.

---

## Comparison with `flash_atten` (MHA)

Both use the same **4-stage Cube–Vec pipeline** and the same kernel source (`../flash_atten/fa_performance_dn_kernel.cpp`):

```
QK (Cube) → P/softmax (Vec) → PV (Cube) → GU (Vec)
```

NSA does **not** fork or modify this pipeline. All NSA-specific logic lives in golden generation and the host driver.

### High-level data flow

```
MHA (flash_atten):
  Q [S0, H]  ×  K [H, S1_full]  →  softmax  →  P × V [S1_full, H]  →  O

NSA (this folder):
  Q [S0, H]  ×  K_cmp [H, S1_cmp]  →  FA  →  O_cmp  ─┐
  Q [S0, H]  ×  K_slc [H, S1_slc]  →  FA  →  O_slc  ─┼→  g₀·O_cmp + g₁·O_slc + g₂·O_win  →  O
  Q [S0, H]  ×  K_win [H, S1_win]  →  FA  →  O_win  ─┘
  ^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  shared Q       branch-specific K/V (shorter S1)  host-side gate fusion (not on NPU yet)
```

### Pipeline / kernel comparison

| Stage | `flash_atten` (MHA) | NSA (this impl) | Code change |
|-------|---------------------|-----------------|-------------|
| **compute_qk** | `Q [Cube_S0, HEAD_SIZE]` × `K [Cube_S1, HEAD_SIZE]` | same per branch | **No** |
| **compute_p** | streaming softmax | same per branch | **No** |
| **compute_pv** | `P @ V [Cube_S1, HEAD_SIZE]` | same per branch | **No** |
| **compute_gu** | running O update | same per branch | **No** |
| **Kernel launches** | 1 × `LaunchTFA` | 3 × `LaunchTFA` | Host only |
| **Output** | single `O` | gated sum of 3 branch `O` | Host only |

**Summary:** NSA reuses the MHA DN kernel **unchanged**. Sparsity comes from **shorter branch sequence lengths** and **different branch K/V tensors**, not from kernel modifications.

### Host and golden differences

| | MHA | NSA |
|---|-----|-----|
| Case tuple | `HEAD,S0,S1,CUBE_S0[,TILE_S1]` | `HEAD,S0,S1,CUBE_S0[,TILE_S1[,BLOCK[,NUM_SEL[,SEL_BLOCK[,WIN[,COMPRESS_DIM]]]]]]` |
| Case name | `case_float_H_{H}_S0_{S0}_S1_{S1}` | `case_float_H_{H}_S0_{S0}_S1_{S1}_B{B}_N{N}_W{W}` |
| Inputs | `q.bin`, `kt.bin`, `v.bin` | `q.bin`, `gates.bin`, per-branch `{cmp,slc,win}_{k,v,kt}.bin` |
| Branch outputs | — | `o_cmp.bin`, `o_slc.bin`, `o_win.bin` (intermediate goldens) |
| Final output | `o.bin` | `o.bin` (gated sum golden), `o_out.bin` (device fused output) |
| Launch API | `LaunchTFA(q, k, v, …)` × 1 | `LaunchTFA(q, k_branch, v_branch, …)` × 3 + host fusion |
| Preprocessing | none | block MLP compress, top-N gather, gate generation (golden) |

### Memory / compute (default case)

Full sequence `S1=512`, branch `S1=128` each (`BLOCK=4`, `NUM_SELECTED=32`, `SELECT_BLOCK=4`, `WINDOW=128`):

| | MHA (one pass) | NSA (three branches) |
|---|----------------|----------------------|
| FA kernel invocations | 1 | 3 |
| Q GM read | `S0 × H × 2 B` | same Q reused (1× read) |
| K/V GM read per pass | `H × S1 × 2 B` + `S1 × H × 2 B` | `H × 128 × 2 B` + `128 × H × 2 B` per branch |
| Effective tokens attended | 512 | 128 + 128 + 128 = 384 (with overlap in design intent) |
| L1 / UB per branch | sized for `S1=512` tiles | sized for `S1_branch=128` (smaller per pass) |

NSA trades **3× FA launch overhead** for **shorter K/V sequences per branch**. Buffer validation (`scripts/validate_buffer_usage.py`) checks UB/L1 against the **worst branch** `S1`.

---

## Comparison with `MLA`

MLA and NSA both extend the A5 FA stack, but they optimize **different bottlenecks**:

| | MHA | MLA | NSA |
|---|-----|-----|-----|
| **Primary goal** | dense attention | KV-cache compression (latent space) | sparse attention (multi-branch) |
| **Kernel change** | baseline | **Yes** — `compute_qk` inner dim | **No** — reuses MHA `LaunchTFA` |
| **QK inner dim** | `head_size` | `kv_latent_dim` | `head_size` |
| **K-side input** | `K [H, S1]` | `c_kv [L, S1]` | branch `K [H, S1_branch]` |
| **V path** | `V [S1, H]` from GM | `V = c_kv @ W_uv` (precomputed in golden) | `V [S1_branch, H]` from GM |
| **FA passes** | 1 | 1 | 3 + host gate fusion |
| **Sparsity mechanism** | none | smaller latent cache | compressed / selected / window K/V |

### Where the code changes

```
MHA:  host (1× LaunchTFA)  +  kernel (QK/PV/GU baseline)

MLA:  host (1× LaunchTMLA)  +  kernel (compute_qk + L1 tiles changed)
                              └─ softmax / PV / GU unchanged

NSA:  host (3× LaunchTFA + gate fusion)  +  golden (compress / select / gate)
      └─ kernel identical to MHA per branch
```

**MLA** changes **inside** the FA kernel (QK matmul and tile shapes).  
**NSA** changes **outside** the FA kernel (which K/V to attend to, and how to combine three FA outputs).

### Design trade-off summary

| Approach | Pros in this repo | Cost |
|----------|-------------------|------|
| **MLA** | Lower KV bandwidth via latent `c_kv`; one kernel launch | New kernel fork; L1 grows when `L > H` |
| **NSA** | No kernel changes; each branch reuses proven MHA DN path | 3× FA launches; compression/selection/gating still on host/golden |

MLA and NSA are **orthogonal**: a production stack could combine MLA-style latent KV with NSA-style sparse branching, but this folder implements NSA on top of the **MHA** kernel only.

---

## Directory layout

```
kernels/manual/a5/NSA/
├── scripts/
│   ├── gen_data.py              # NSA golden (compress, select, 3 branches, gates)
│   ├── generate_cases.py        # Case config with branch S1 derivation
│   └── validate_buffer_usage.py # UB/L1 checks per branch
├── main.cpp                     # 3× LaunchTFA + host gate fusion
├── run.sh
├── CMakeLists.txt               # links ../flash_atten/fa_performance_dn_kernel.cpp
└── README.md
```

There is no separate `nsa_performance_dn_kernel.cpp`; the DN kernel is shared with [`../flash_atten`](../flash_atten).

---

## Case format

```
HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1[,BLOCK_SIZE[,NUM_SELECTED[,SELECT_BLOCK_SIZE[,WINDOW[,COMPRESS_DIM]]]]]]
```

Derived branch lengths:

- `S1_CMP = S1 / BLOCK_SIZE`
- `S1_SLC = NUM_SELECTED × SELECT_BLOCK_SIZE`
- `S1_WIN = WINDOW_SIZE`

Each branch `S1_*` must be divisible by `TILE_S1`.

Example:

```bash
source /usr/local/Ascend/cann_9b2/cann/bin/setenv.bash
cd kernels/manual/a5/NSA
bash run.sh -r sim -v Ascend950PR_9599 --cases "128,128,512,128,128,4,32,4,128,256" -p 2 -m 1
```

Defaults: `BLOCK_SIZE=4`, `NUM_SELECTED=32`, `SELECT_BLOCK_SIZE=4`, `WINDOW=128`, `COMPRESS_DIM=256` → each branch `S1=128` with `TILE_S1=128`.

---

## Scope and follow-ups

| Component | Status |
|-----------|--------|
| Golden: block compress + top-N select + 3 branch FA + gates | Implemented |
| A5: 3× `LaunchTFA` + host gate fusion | Implemented (reuses `../flash_atten` DN kernel) |
| On-NPU block compression MLP | Not yet (precomputed in golden) |
| On-NPU top-N block selection | Not yet (pre-gathered K/V in golden) |
| On-NPU gate fusion (Vec) | Not yet (host FP32 fusion) |
| Fused single-kernel NSA | Future work |
| NSA + MLA (latent KV + sparse branches) | Future work |

---

## References

- [`../flash_atten`](../flash_atten) — MHA DN kernel reused per branch
- [`../MLA`](../MLA) — latent-attention variant (different QK path)
- [`Projects/attention/pytorch/NSA.py`](../../../../../../attention/pytorch/NSA.py) — PyTorch NSA reference
- [`Projects/attention/skills/npu/A5/a5-fa-implementation.md`](../../../../../../attention/skills/npu/A5/a5-fa-implementation.md) — MHA/GQA/MLA/DSA kernel reuse analysis
