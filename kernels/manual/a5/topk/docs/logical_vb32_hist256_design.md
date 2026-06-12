# Logical `VLxN` Register — Design Doc

**Status:** implemented for `Vb32Hist256` in
[scripts/numpy_radix_topk_intrinsic_map.py](../scripts/numpy_radix_topk_intrinsic_map.py).
**Scope:** A5 NPU vector pipe (CCE intrinsics: `chistv2`, `vcvt PART_{EVEN,ODD}`,
`vadd`, `vsts INTLV_B32`). Generalises to any value that does not fit in a
single VL register but is naturally a small fan-out of VL-sized physical regs.

---

## 1. Problem

The A5 cumulative-histogram primitive `chistv2` returns 128 u16 bins per call
(`Bin_N0` → bins `0..127`, `Bin_N1` → bins `128..255`). To accumulate the bins
in u32 across many input repeats, each `chistv2` output has to be widened by
`vcvt PART_EVEN` and `vcvt PART_ODD`, producing **four** VL_B32 (64-lane u32)
physical registers per logical 256-bin histogram:

```
chistv2 Bin_N0 ──► vb16(128) ──┬─ vcvt PART_EVEN ─► n0_even : VL_B32 u32   (bins 0,2,…,126)
                               └─ vcvt PART_ODD  ─► n0_odd  : VL_B32 u32   (bins 1,3,…,127)
chistv2 Bin_N1 ──► vb16(128) ──┬─ vcvt PART_EVEN ─► n1_even : VL_B32 u32   (bins 128,130,…,254)
                               └─ vcvt PART_ODD  ─► n1_odd  : VL_B32 u32   (bins 129,131,…,255)
```

Storing those 4 regs back to a contiguous 256-u32 UB region requires two
`vsts INTLV_B32` ops, one per `{even, odd}` pair, each interleaving lanes at
stride 2.

The straight transliteration ends up with eight named registers, four `vadd`s,
two `vsts INTLV_B32`s, plus tail-predicate bookkeeping — duplicated at every
histogram call site. The intent ("accumulate one 256-bin u32 cumulative
histogram of a `vlds`-loaded source") is buried under interleave plumbing that
is mechanically inferable from the layout.

## 2. Concept — a logical `VLxN` array

Model the histogram as **one** logical register of `256 × u32` whose physical
backing is `N = 4` VL_B32 regs. Capture the relationship between *logical bin
index* and *physical (reg, lane)* as a small piece of **stride metadata**:

```
logical bin b ∈ [0, 256)
half  = b // 128            ∈ {0, 1}        → selects {n0_*, n1_*}      (Bin_N{0,1})
parity = (b % 128) % 2      ∈ {0, 1}        → selects {*_even, *_odd}    (PART_{EVEN,ODD})
lane   = (b % 128) // 2     ∈ [0, VL_B32)   → physical lane index
```

The four physical regs are therefore a **two-axis strided view** of the logical
256-bin array, with axes `(half, parity)` and lane stride 1 inside each reg.

Given this metadata, the compiler/wrapper can *infer* the right instructions:

| Access pattern | Inferred ops |
|---|---|
| `.zero()` | `vbr u32 0`, ×4 (one per physical reg) |
| Per-lane elementwise op on the whole logical reg (e.g. `vadd`) | Same op, ×4 (one per physical reg) under a full b32 predicate |
| Per-half reduction op (e.g. `chistv2` writing into the reg) | `chistv2 Bin_N0` then `chistv2 Bin_N1`, each followed by `vcvt PART_EVEN` + `vcvt PART_ODD` + `vadd` to the matching pair |
| **Contiguous store** to a 256-u32 UB region | `vsts INTLV_B32(*_even, *_odd)` ×2 — one per half |
| **Contiguous read** view (Python only, for verification) | `np.lib.stride_tricks.as_strided`-style interleave of the four regs |

In other words: **the stride metadata is sufficient to choose between the
"plain" b32 mode and the `INTLV_B32`/`PART_{EVEN,ODD}` interleaved mode at
every consumer.**

## 3. Layout & strides — diagram

Logical view (`.as_contiguous()` returns this):

```
bin index :  0   1   2   3   4   5   …  126 127 128 129 130 131 … 254 255
backing   :  Ee  Eo  Ee  Eo  Ee  Eo  …  Ee  Eo  Fe  Fo  Fe  Fo  … Fe  Fo
where      E = n0_*,  F = n1_*,   e = *_even,  o = *_odd
```

Physical view (4 × VL_B32 = 4 × 64 u32 regs):

```
n0_even[0..63]  →  logical bins {0, 2, …, 126}
n0_odd [0..63]  →  logical bins {1, 3, …, 127}
n1_even[0..63]  →  logical bins {128, 130, …, 254}
n1_odd [0..63]  →  logical bins {129, 131, …, 255}
```

This is exactly the layout `vsts INTLV_B32` writes to memory: interleaving the
even/odd reg lane-by-lane into a stride-1 region.

## 4. API

```python
@dataclass
class Vb32Hist256:
    n0_even: np.ndarray   # VL_B32 u32 — bins 0,2,…,126
    n0_odd:  np.ndarray   # VL_B32 u32 — bins 1,3,…,127
    n1_even: np.ndarray   # VL_B32 u32 — bins 128,130,…,254
    n1_odd:  np.ndarray   # VL_B32 u32 — bins 129,131,…,255

    @classmethod
    def zero(cls) -> "Vb32Hist256": ...         # vbr ×4
    def as_contiguous(self) -> np.ndarray: ...  # logical 256-u32 view

def chist_accumulate(acc: Vb32Hist256,
                     src_u8_vec: np.ndarray,   # one VL_B8 source register
                     pred256: np.ndarray) -> Vb32Hist256:
    """chistv2(N0)+chistv2(N1) + vcvt(EVEN,ODD)×2 + vadd×4, hidden."""

def vsts_hist256(ub_dst_u32: np.ndarray,
                 off_u32: int,
                 acc: Vb32Hist256,
                 pred256: np.ndarray | None = None) -> None:
    """Auto-emits two `vsts INTLV_B32` ops at offsets 0 and 2*VL_B32."""
```

Design properties:

* **Functional update.** `chist_accumulate` returns a new `Vb32Hist256` —
  matches CCE's destination-register semantics and keeps the type immutable
  from the caller's point of view.
* **Single predicate input.** A b8 predicate (256 bools) is passed in; the
  wrapper derives the b16/b32 full-true predicates for the intermediate
  `vcvt`/`vadd` ops. The user never names them.
* **No leak of physical register count.** The arity "4" only shows up in the
  dataclass fields and the two `vsts_intlv_b32` calls inside `vsts_hist256`.
* **`.as_contiguous()` mirrors `vsts_hist256`.** What you'd see in Python
  is byte-equal to what ends up in UB after the store. Useful as a debug
  oracle and as the basis for unit tests.

## 5. Inference rules — how "logical access" maps to instructions

The wrapper functions encode three inference rules. Each rule is mechanical;
none requires user input beyond the data and a predicate.

### Rule R1 — Elementwise per-lane ops fan out across physical regs

If an op is per-lane and dtype-uniform across the logical array
(e.g. `vadd`, `vmin`, `vsub`, `vmul`, `vmov`), emit the op once per physical
reg under a full b32 predicate. No interleaving is involved.

This is what `chist_accumulate` does for its four `vadd_u32` calls.

### Rule R2 — Per-half producer maps to `Bin_N{0,1}` then EVEN/ODD widen

If a producer naturally writes half the logical reg at a time
(`chistv2 Bin_N0` / `Bin_N1`), call it twice and feed each output through
`vcvt PART_EVEN` and `vcvt PART_ODD` to land into the matching `{n*_even,
n*_odd}` slots.

This is what `chist_accumulate` does for its two `chistv2` calls.

### Rule R3 — Contiguous UB access infers `vsts INTLV_B32` (store) / DINTLV (load)

If the consumer of the logical reg needs a **stride-1 view in bin order** in
UB memory:

* **Store** — infer `vsts INTLV_B32(*_even, *_odd)` per half (one per pair).
* **Load** — would infer `vlds DINTLV_B32(*_even, *_odd)` per half (symmetric;
  not used in topk so not implemented yet — see §8).

The wrapper iterates over `(half_index, pair)` and emits the corresponding
intrinsic without the caller naming `INTLV_B32`.

This is what `vsts_hist256` does.

## 6. Worked example — `t_histogram_simd` before/after

**Before** (explicit chain, ~30 lines per row):

```python
preg_all_b16 = pset_b16_all()
preg_all_b32 = pset_b32_all()

vb32_n0_even_inc = vbr(elem_bytes=4, value=0, dtype=np.uint32)
vb32_n0_odd_inc  = vbr(elem_bytes=4, value=0, dtype=np.uint32)
vb32_n1_even_inc = vbr(elem_bytes=4, value=0, dtype=np.uint32)
vb32_n1_odd_inc  = vbr(elem_bytes=4, value=0, dtype=np.uint32)

for c in range(repeat_times_per_row):
    ...
    vb16_BIN_N0 = chistv2(src_for_hist, pred0, bin_part=0)
    vb16_BIN_N1 = chistv2(src_for_hist, pred1, bin_part=1)
    vb32_n0_even = vcvt_b16_to_b32(vb16_BIN_N0, preg_all_b16, "EVEN")
    vb32_n0_odd  = vcvt_b16_to_b32(vb16_BIN_N0, preg_all_b16, "ODD")
    vb32_n1_even = vcvt_b16_to_b32(vb16_BIN_N1, preg_all_b16, "EVEN")
    vb32_n1_odd  = vcvt_b16_to_b32(vb16_BIN_N1, preg_all_b16, "ODD")
    vb32_n0_even_inc = vadd_u32(vb32_n0_even_inc, vb32_n0_even, preg_all_b32, "zeroing")
    vb32_n0_odd_inc  = vadd_u32(vb32_n0_odd_inc,  vb32_n0_odd,  preg_all_b32, "zeroing")
    vb32_n1_even_inc = vadd_u32(vb32_n1_even_inc, vb32_n1_even, preg_all_b32, "zeroing")
    vb32_n1_odd_inc  = vadd_u32(vb32_n1_odd_inc,  vb32_n1_odd,  preg_all_b32, "zeroing")

vsts_intlv_b32(bin_count[r], 0,            vb32_n0_even_inc, vb32_n0_odd_inc, preg_all_b32)
vsts_intlv_b32(bin_count[r], 2 * VL_B32,   vb32_n1_even_inc, vb32_n1_odd_inc, preg_all_b32)
```

**After** (logical register + inferred modes, ~8 lines per row):

```python
hist_acc = Vb32Hist256.zero()
for c in range(repeat_times_per_row):
    ...
    hist_acc = chist_accumulate(hist_acc, src_for_hist, pred)
vsts_hist256(bin_count[r], 0, hist_acc)
```

The four named accumulators, the two `vcvt PART_*` pairs, the four `vadd`s,
and the two `vsts INTLV_B32`s all collapse into the rules R1–R3 evaluated by
`chist_accumulate` and `vsts_hist256`. The user code mentions neither EVEN/ODD
nor INTLV_B32.

## 7. Why this is "stride metadata", not "yet another wrapper"

The rules above are **derived from a single declarative description** of the
logical layout:

```
Vb32Hist256:
  logical_shape   = (256,)
  logical_dtype   = u32
  physical_regs   = 4
  physical_dtype  = u32
  physical_lanes  = VL_B32 (= 64)
  axes:
    half   : split logical bins into halves of 128; selects (n0_*, n1_*);
             produced by chistv2 Bin_N{0,1}.
    parity : stride-2 interleave within each half; selects (*_even, *_odd);
             produced by vcvt PART_{EVEN, ODD};
             reconstructed by vsts INTLV_B32 (and read back by vlds DINTLV_B32).
```

Any future op on the same logical type only needs to consult this metadata to
pick its physical realisation. For instance, `vrmin` over the whole logical
256 bins would emit four `vrmin` per physical reg plus a 4-lane scalar reduce;
a `vgather` indexed by logical bin index would decode `(half, parity, lane)`
and emit one indexed read per physical reg.

That is the sense in which the abstraction "infers compute and store modes from
strides." The user describes the computation against the **logical** array;
the inference layer picks `INTLV_B32` / `PART_{EVEN,ODD}` whenever the access
pattern requires stride-1 logical bin order.

## 8. Generalisation — `LogicalVLxN`

The pattern is not unique to histograms. Any value of size `N × VL_LANE` for
small `N` can be a logical register with the same recipe:

```python
@dataclass
class LogicalVLxN:
    physical: tuple[np.ndarray, ...]    # N VL-sized regs of the same dtype
    layout:   LayoutDescriptor          # axes + producer/consumer rules
```

`LayoutDescriptor` carries the rules from §5: which axis corresponds to
which CCE mode (`INTLV`, `DINTLV`, `BRC`, `PART_EVEN/ODD`, `Bin_N{0,1}`, …)
and the natural fan-out for elementwise ops.

Concrete near-term candidates:

* **`Vb16Hist256`** (when bin counts fit in u16): `N = 2`, only the `half` axis
  remains; `vcvt` collapses out; stores via two plain `vsts B16`.
* **`Vb32Range1024`** (4×VL_B32 contiguous, no parity axis): `N = 4`, only the
  "chunk" axis; stores via four plain `vsts B32` — already used implicitly by
  the per-VL_B32 chunk loops in phase1/3.
* **`Vb32GatherBuf`** (load via `vlds DINTLV_B32` for `vgather` inputs): the
  symmetric load path for Rule R3.

The Top-K pipeline only needs `Vb32Hist256` today, but lifting the other two
out of the per-phase chunk loops would let phase1/3 read like

```python
acc = Vb32Range1024.from_ub(global_bin_count_msb, 0)
acc = vadd(acc, Vb32Range1024.from_ub(bin_count[r], 0))
vsts(global_bin_count_msb, 0, acc)
```

instead of the explicit `for c in range(4): vlds_u32 / vadd_u32 / vsts_u32`
loop.

## 9. Op classification — which APIs support `VLxN` natively, and when to insert `.contiguous()`

A `VLxN` value is just **`N` physical VL registers plus a `LayoutDescriptor`**.
Whether the next op can be lowered without first materialising the contiguous
logical view depends entirely on (a) the op's access pattern and (b) the
layout descriptor's axes. The compiler / 1:1-mapping library classifies every
op into one of three buckets and either propagates layout, rewrites the
op into the matching interleaved mode, or **automatically inserts a
`.contiguous()` materialisation** before the op.

### 9.1 Three categories of ops

| Category | Definition | Lowering | Output layout |
|---|---|---|---|
| **A. Native-strided (elementwise / per-lane)** | Op is per-lane, dtype-uniform, independent across logical indices. Touches every logical lane through the *same* physical-reg coordinates. | Fan out the op once per physical reg under a full b-stride predicate. (Rule R1.) | **Same layout as input** — metadata propagates unchanged. |
| **B. Mode-rewritable (interleave-aware modes)** | Op has a dedicated CCE *mode* that matches one of the layout axes (e.g. `INTLV_B32`, `DINTLV_B{8,16,32}`, `BRC_B8`, `PART_{EVEN,ODD}`, `Bin_N{0,1}`). | Decompose into one op per axis-fan-out, picking the matching mode. (Rules R2, R3.) | **Layout consumed** (store / reduce) or **layout produced** (load / widen). |
| **C. Contiguous-required** | Op needs stride-1 logical-index access and has *no* mode covering the layout's axes (e.g. arbitrary-index `vgather`, cross-axis reduction, a producer/consumer whose natural stride disagrees with the descriptor). | Insert an automatic `.contiguous()` to materialise the logical view in UB / a temporary tile, then run the plain stride-1 op. | **Plain VL / VLxN linear layout** (no parity, no half axes). |

The library catalogues each CCE intrinsic with the **set of layout axes it
can absorb natively**. Examples for the A5 ops touched by Top-K:

| Intrinsic | Category | Absorbs which axes |
|---|---|---|
| `vbr`, `vmov`, `vadd`, `vsub`, `vmin`, `vmul`, `vor`, `vand`, `vshls`, `vsels`, `vsel`, `vcmps` | **A** | any (per-lane) |
| `vsts INTLV_B{8,16,32}` (store) | **B** | `parity` (stride-2 within a half) |
| `vlds DINTLV_B{8,16,32}` (load) | **B** | `parity` |
| `vlds BRC_B8` | **B** | `broadcast` (collapses an axis to 1) |
| `vcvt PART_{EVEN,ODD}` | **B** | `parity` (produces it from a u16 reg) |
| `chistv2 Bin_N{0,1}` | **B** | `half` (produces it from a u8 reg) |
| `vrmin`, `vrmax` (full-reg reduction) | **C** within a single VL reg; **C** *across* physical regs of a `VLxN` (needs cross-reg reduce → materialise) | — |
| `vgather` with arbitrary indices | **C** | — (needs stride-1 table in UB) |
| `TROWMIN`, `TCONCAT_IMPL`, `TSTORE` (tile-level ops) | **C** | — (consume contiguous tiles by definition) |

### 9.2 Metadata carried in the `VLxN` value

Every `VLxN` value carries the descriptor in its type. A minimal sketch:

```python
@dataclass(frozen=True)
class LayoutAxis:
    name:      str            # e.g. "half", "parity", "chunk"
    cardinality: int          # how many physical regs this axis fans out to
    mode:      str | None     # producer/consumer CCE mode, e.g. "INTLV_B32",
                              # "PART_EVEN/ODD", "Bin_N0/N1", "BRC_B8", None
    stride_in_logical: int    # element stride in the logical view (1 for chunk,
                              # 2 for parity, 128 for half@u32 256-bin, …)

@dataclass(frozen=True)
class LayoutDescriptor:
    logical_shape: tuple[int, ...]
    logical_dtype: np.dtype
    phys_dtype:    np.dtype
    phys_lanes:    int                # VL_B{8,16,32}
    axes:          tuple[LayoutAxis, ...]

    @property
    def is_contiguous(self) -> bool:    # only axes with mode=None (chunk) remain
        return all(a.mode is None for a in self.axes)
```

For `Vb32Hist256` this is

```python
LayoutDescriptor(
    logical_shape=(256,), logical_dtype=u32, phys_dtype=u32, phys_lanes=64,
    axes=(LayoutAxis("half",   2, mode="Bin_N0/N1 ↔ INTLV_B32 across halves", 128),
          LayoutAxis("parity", 2, mode="PART_EVEN/ODD ↔ INTLV_B32", 2)),
)
```

For the `Vb32Range1024` (4×VL_B32 contiguous, no interleave):

```python
LayoutDescriptor(
    logical_shape=(256,), logical_dtype=u32, phys_dtype=u32, phys_lanes=64,
    axes=(LayoutAxis("chunk", 4, mode=None, stride_in_logical=64),),
)   # is_contiguous == True
```

### 9.3 Propagation & rewriting rules (what the compiler does at each call site)

Given an op and an input `VLxN v` with descriptor `L`:

1. **Lookup the op's absorbable-axis set `A_op`.**
2. **If the op is Category A:** emit one physical op per element of
   `∏ axis.cardinality` of `L`, copying `L` to the result. (Layout passes
   through unchanged.)
3. **If the op is Category B:** find the axis `a ∈ L.axes` whose `mode` matches
   an entry in `A_op`. Emit one op per element of the *other* axes, instantiating
   the matching mode for `a`. The result either:
    * **consumes** `a` (e.g. `vsts INTLV_B32` collapses the parity axis → result
      lives in UB with `parity` removed from its descriptor), or
    * **produces** `a` (e.g. `chistv2` produces `half`; `vcvt PART_*` produces
      `parity`).
4. **If the op is Category C, or no `a ∈ L.axes` matches anything in `A_op`:**
   **insert `v.contiguous()`** before the op. The contiguous form has a flat
   `chunk`-only descriptor (`is_contiguous == True`) and can feed any
   stride-1 consumer.

The check at step 4 is what makes the system safe: a user can sprinkle any
op into the IR, and the worst the compiler will do is materialise a
`.contiguous()` view; it will never silently mis-lower an interleave.

### 9.4 What `.contiguous()` lowers to

For a `VLxN` value that lives in **registers**:

* For each axis with `mode ∈ {INTLV_B{8,16,32}, PART_{EVEN,ODD}, Bin_N{0,1}}`,
  emit the corresponding store-with-mode to a UB scratch region, then `vlds`
  the same region as a flat `VLxN` (chunk-only). For the `Vb32Hist256` case
  this is exactly `vsts_hist256(scratch, 0, v)` followed by four `vlds_u32`
  of `scratch[0..256]`.
* For axes with `mode = BRC_B8`, no materialisation is required; the broadcast
  axis collapses to 1 and the result is already contiguous.
* For an axis with `mode = None` (i.e. `chunk`), nothing to do for that axis.

For a `VLxN` value that **already lives in UB** with an interleaved layout
(rare; arises only if a producer was forced to store eagerly), `.contiguous()`
is a no-op marker: the data in UB is already in logical bin order if the
producer used `vsts INTLV_B32` — which is precisely the post-condition of
Rule R3.

**Cost model.** Insertion of `.contiguous()` costs the `N` stores + `N`
loads required by the materialisation. The compiler should:

* **Never insert `.contiguous()` redundantly** — track an `is_contiguous` bit
  in the descriptor and skip if already true.
* **Hoist** `.contiguous()` out of loops whenever the source is loop-invariant
  (e.g. the Phase-1 chist output staying in UB across all Phase-2 readers).
* **Fuse** consecutive contiguous-required consumers so they share a single
  materialisation.

### 9.5 Worked propagation example — `Vb32Hist256` lifecycle in Top-K

```
chist_accumulate(...)      # Category B (Bin_N{0,1} + PART_{EVEN,ODD})  produces (half, parity)
       ↓ Vb32Hist256
vadd(acc, other_hist)      # Category A (elementwise)                   layout passes through
       ↓ Vb32Hist256
vsts_hist256(ub, 0, acc)   # Category B (INTLV_B32)                     consumes parity → UB stride-1
       ↓ UB[256 u32]                                                    is_contiguous == True
TROWMIN(...) / vgather(...) # Category C — reads stride-1 UB             no .contiguous() needed
                                                                        (already contiguous)
```

Hypothetical: if the user wrote `vrmin(hist_acc)` directly on the logical reg
(reduce all 256 bins to the global min), the compiler would observe
`vrmin ∉ A`, no matching mode in `B`, so it would **auto-insert
`hist_acc.contiguous()` first**, producing a `Vb32Range1024`, on which the
loop-fanned-out `vrmin` (four per-reg minima + a 4-lane scalar reduce) lowers
trivially.

### 9.6 API contract summary

For library/compiler authors implementing the `VLxN` interface:

* **Every CCE intrinsic must declare its category and absorbable axes** in a
  static op-table. The wrapper layer uses this table to pick A / B / C
  lowering at every call site.
* **Every `VLxN` constructor produces a value with a fully populated
  `LayoutDescriptor`.** Producers in Category B set the axis they just
  introduced; producers in Category A keep their input's descriptor.
* **`.contiguous()` is the only legal escape hatch.** It is the sole API
  permitted to drop layout metadata. All other layout-changing transitions go
  through a Category-B op.
* **Users never spell `INTLV_B32` / `DINTLV_B32` / `PART_{EVEN,ODD}` /
  `Bin_N{0,1}` directly** — they appear only inside the op-table entries and
  in the wrapper bodies for Category B ops.

## 10. Verification

Implemented unit checks (run as a one-liner after the refactor):

1. **Layout equivalence.** For random `src`:
   `vsts_hist256(ub, 0, chist_accumulate(zero(), src, p_all)) == acc.as_contiguous()` ✓
2. **Algorithmic equivalence.** Same input:
   `ub == np.cumsum(np.bincount(src, minlength=256)).astype(np.uint32)` ✓
3. **Multi-repeat accumulation.** 8 repeats × VL_B8 random bytes:
   final `ub` matches `cumsum(bincount(big, 256))` ✓
4. **End-to-end.** `radix_topk` matches the golden multiset on
   seeds `{1241200609, 2203936584, 191132090}` and `--const 0x1234` ✓

## 11. Non-goals / open questions

* **No code generation.** This doc describes the abstraction in Python (the
  semantic spec). The real C++ THistogram is unchanged; the wrapper is a
  modelling device that documents *which* CCE modes are mechanical from the
  layout and which are not.
* **Per-half predicate.** `chist_accumulate` takes a single `pred256` for both
  `Bin_N0` and `Bin_N1`. The current `t_histogram_simd` always passes the same
  predicate to both halves (lane-tail-trim or vcmp_eq filter applied uniformly),
  so this is sound. A future caller that needs distinct N0/N1 predicates would
  need either a `chist_accumulate(acc, src, pred_n0, pred_n1)` overload or two
  separate `chist_half_accumulate` calls.
* **Load path.** Rule R3 currently only handles the store direction
  (`vsts INTLV_B32`). The symmetric `vlds DINTLV_B32` load — needed if we
  ever want to *resume* an in-UB 256-bin histogram into a `Vb32Hist256` — is
  not implemented yet.
* **Dtype variety.** `Vb32Hist256` is u32-specific. Variants for u16 / fp32
  would follow the same recipe with adjusted `vcvt` rules.

---

### Appendix — code pointers

* `Vb32Hist256`, `chist_accumulate`, `vsts_hist256`:
  [scripts/numpy_radix_topk_intrinsic_map.py](../scripts/numpy_radix_topk_intrinsic_map.py)
  (search for `# Vb32Hist256 — logical 256-bin u32 cumulative-histogram register.`).
* Consumer: `t_histogram_simd` in the same file.
* Reference C++ implementation: `pto::THistogram<TileDst, TileSrc, TileIdx, isMSB>`
  in `include/pto/npu/a5/THistogram.hpp`.
