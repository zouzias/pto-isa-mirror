# PyPTO PA4 (Paged Attention) Pass-by-Pass Compilation Analysis

This document provides a comprehensive analysis of each PyPTO compilation pass for a Paged Attention kernel, including pass source code analysis and highlighting whether changes occur in the PA4 case.

**Source Repository:** [hengliao1972/pypto-lib](https://github.com/hengliao1972/pypto-lib/tree/main/examples/pa4_build/passes_dump)

---

## Table of Contents

1. [Pass Pipeline Overview](#pass-pipeline-overview)
2. [Pass 00: Frontend](#pass-00-frontend)
3. [Pass 01: UnrollLoops](#pass-01-unrollloops)
4. [Pass 02: ConvertToSSA](#pass-02-convertossa)
5. [Pass 03: FlattenCallExpr](#pass-03-flattencallexpr)
6. [Pass 04: SplitChunkedLoops](#pass-04-splitchunkedloops)
7. [Pass 05: InterchangeChunkLoops](#pass-05-interchangechunkloops)
8. [Pass 06: RunVerifier](#pass-06-runverifier)
9. [Pass 07: OutlineIncoreScopes](#pass-07-outlineincorescopes)
10. [Pass 08: ExpandMixedKernel](#pass-08-expandmixedkernel)
11. [Pass 09: ConvertTensorToBlockOps](#pass-09-converttensorblockops)
12. [Pass 10: InitMemRef](#pass-10-initmemref)
13. [Pass 11: MemoryReuse](#pass-11-memoryreuse)
14. [Pass 12: InsertSync](#pass-12-insertsync)
15. [Pass 13: AllocateMemoryAddr](#pass-13-allocatememoryaddr)
16. [Architecture Mapping](#architecture-mapping)

---

## Pass Pipeline Overview

```
00_frontend.py              ← Original DSL code
       │
       ▼
01_after_UnrollLoops.py     ← Compile-time loop unrolling
       │
       ▼
02_after_ConvertToSSA.py    ← SSA conversion with iter_args
       │
       ▼
03_after_FlattenCallExpr.py ← Three-address code conversion
       │
       ▼
04_after_SplitChunkedLoops.py   ← Split parallel loops by chunk
       │
       ▼
05_after_InterchangeChunkLoops.py ← Reorder chunk loops
       │
       ▼
06_after_RunVerifier.py     ← IR verification
       │
       ▼
07_after_OutlineIncoreScopes.py ← Extract InCore kernel + Orchestration
       │
       ▼
08_after_ExpandMixedKernel.py   ← Split into AIC + AIV + TPUSH/TPOP
       │
       ▼
09_after_ConvertTensorToBlockOps.py ← Tensor ops → Tile/Block ops
       │
       ▼
10_after_InitMemRef.py      ← Add MemRef annotations (DDR, sizes)
       │
       ▼
11_after_MemoryReuse.py     ← Buffer lifetime analysis & reuse
       │
       ▼
12_after_InsertSync.py      ← Pipeline synchronization insertion
       │
       ▼
13_after_AllocateMemoryAddr.py ← Final address allocation
```

---

## Pass 00: Frontend

**Source File:** N/A (Python DSL input)

**Purpose:** Original user-written PyPTO DSL code representing the Paged Attention algorithm.

**Key Features in PA4:**
- `pl.parallel(0, 64, 1, chunk=8)` - Parallel loops with chunking
- `pl.range(0, bn_this_batch, 1)` - Dynamic range loops
- `pl.tensor.matmul()` - Matrix multiplication operations
- `pl.tensor.row_max()`, `pl.tensor.row_sum()` - Reduction operations
- `pl.tensor.assemble()` - Tensor assembly (scatter writes)
- Online softmax algorithm with `mi_update`, `li_update` state

**PA4 IR Characteristics:**
```python
for b_idx in pl.parallel(0, 64, 1, chunk=8):
    for q_idx in pl.parallel(0, 4, 1, chunk=2):
        # Mutable state variables
        oi = pl.tensor.create(...)
        li_update = pl.tensor.create(...)
        mi_update = pl.tensor.create(...)
        
        for bn in pl.range(0, bn_this_batch, 1):
            # Attention computation with online softmax
            sij = pl.tensor.matmul(qi, kj, b_trans=True)
            mi = pl.tensor.row_max(scaled)
            # ... update mi_update, li_update, oi ...
```

---

## Pass 01: UnrollLoops

**Source File:** `src/ir/transforms/unroll_loops_pass.cpp`

**Purpose:** Expands loops with compile-time constant bounds marked with `ForKind::Unroll` into a sequence of cloned loop bodies, substituting the loop variable with each iteration's constant value.

**Key Logic from Source:**
```cpp
// Maximum iterations allowed for compile-time unrolling
constexpr int64_t kMaxUnrollIterations = 1024;

// LoopUnrollMutator expands ForStmt nodes with ForKind::Unroll into
// a SeqStmts of cloned bodies, substituting the loop variable with each
// iteration's constant value.
class LoopUnrollMutator : public IRMutator {
    // Each unrolled iteration gets fresh Var objects for definition sites
    // to ensure structural equality works correctly.
};
```

**Transformation:**
- Finds `ForStmt` with `ForKind::Unroll`
- Extracts compile-time constant start/stop/step
- Clones loop body N times, substituting loop variable with constants
- Creates fresh Var objects per iteration for SSA correctness

| PA4 Status | **NO CHANGE** |
|------------|---------------|
| Reason | PA4 has no `ForKind::Unroll` loops; all loops are `pl.parallel` or `pl.range` with dynamic bounds (`bn_this_batch`) |

---

## Pass 02: ConvertToSSA ⭐

**Source File:** `src/ir/transforms/convert_to_ssa_pass.cpp`

**Purpose:** Converts mutable variable semantics to Static Single Assignment (SSA) form using `init_values` and `pl.yield_()` for loop-carried dependencies.

**Key Logic from Source:**
```cpp
// AssignmentCollector: Collects all assigned variable base names in a statement
// Used to pre-analyze loop bodies to find which outer variables are modified,
// allowing us to create iter_args before visiting the body.
class AssignmentCollector : public IRVisitor {
    std::set<std::string> assigned_vars;
    
    void VisitStmt_(const AssignStmtPtr& op) override {
        assigned_vars.insert(GetBaseName(op->var_->name_));
    }
};
```

**Transformation:**
1. Pre-analyze loop bodies to identify modified variables
2. Create `iter_args` for loop-carried state
3. Rename all variable definitions to unique SSA names (`var_0`, `var_1`, ...)
4. Insert `pl.yield_()` at loop exits to propagate updated values
5. Maps to MLIR `scf.for` with `iter_args`

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | All mutable variables (`oi`, `li_update`, `mi_update`, `out`) converted to SSA with `init_values` and `yield` |

**Before:**
```python
for bn in pl.range(0, bn_this_batch, 1):
    mi_update = mi
    li_update = li
    oi = oi_tmp
    out = pl.tensor.assemble(out, dst, ...)
```

**After:**
```python
for bn_0, (li_update_iter_1, mi_update_iter_1, oi_iter_1, out_iter_5) in pl.range(
    0, bn_this_batch_0, 1, 
    init_values=(li_update_0, mi_update_0, oi_0, out_iter_3)
):
    mi_update_3 = mi_0
    li_update_3 = li_0
    oi_3 = oi_tmp_0
    out_7 = pl.tensor.assemble(out_iter_5, dst_0, ...)
    li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(li_update_3, mi_update_3, oi_3, out_7)
```

---

## Pass 03: FlattenCallExpr

**Source File:** `src/ir/transforms/flatten_call_expr_pass.cpp`

**Purpose:** Flattens nested call expressions into three-address code by extracting nested calls into temporary variables.

**Key Logic from Source:**
```cpp
// Mutator that flattens nested call expressions into three-address code
// This pass ensures that:
// 1. Call arguments cannot be calls
// 2. If conditions cannot be calls
// 3. For loop ranges (start/stop/step) cannot be calls
// 4. Binary/unary expression operands cannot be calls
// 
// Nested calls are extracted into temporary variables and inserted as
// AssignStmt before the statement containing the nested call.
class FlattenCallExprMutator : public IRMutator { ... };
```

**Transformation:**
- `f(g(x))` → `tmp = g(x); f(tmp)`
- Ensures all call arguments are simple variables or constants

| PA4 Status | **NO CHANGE** |
|------------|---------------|
| Reason | PA4 already uses flattened call style; no nested calls like `pl.tensor.add(pl.tensor.mul(a, b), c)` |

---

## Pass 04: SplitChunkedLoops

**Source File:** `src/ir/transforms/split_chunked_loops_pass.cpp`

**Purpose:** Splits `pl.parallel(..., chunk=N)` loops into outer (orchestration) and inner (incore) loops.

**Key Logic from Source:**
```cpp
// Extract compile-time integer for chunk size
static int64_t GetConstIntValue(const ExprPtr& expr, const std::string& what);

// Collect all AssignStmt var_ (DEF sites) from a statement tree.
// When the body is visited multiple times (inner + remainder), the same
// VarPtr would appear as a DEF in both, violating SSA.
static void CollectDefVars(const StmtPtr& stmt, std::vector<VarPtr>& result);
```

**Transformation:**
```
pl.parallel(0, 64, 1, chunk=8)  
    → outer: pl.range(0, 8, 1)      # 64/8 = 8 iterations
    → inner: pl.parallel(0, 8, 1)   # chunk size
```

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | `pl.parallel(0, 64, chunk=8)` split into outer `pl.range(0, 8)` + inner `pl.parallel(0, 8)` |

**Before:**
```python
for b_idx in pl.parallel(0, 64, 1, chunk=8):
    for q_idx in pl.parallel(0, 4, 1, chunk=2):
```

**After:**
```python
for b_idx_0_out in pl.range(0, 8, 1, ...):      # 64/8 = 8 outer
    for q_idx_0_out in pl.range(0, 2, 1, ...):  # 4/2 = 2 outer
        for b_idx_0_in in pl.parallel(0, 8, 1, ...):  # chunk=8 inner
            for q_idx_0_in in pl.parallel(0, 2, 1, ...):  # chunk=2 inner
```

---

## Pass 05: InterchangeChunkLoops

**Source File:** `src/ir/transforms/interchange_chunk_loops_pass.cpp`

**Purpose:** Reorders loop nesting to ensure outer loops are orchestration-level (`pl.range`) and inner loops are incore-level (`pl.parallel`).

**Key Logic from Source:**
```cpp
// Check if a statement body contains a ScopeStmt(InCore).
static bool ContainsInCoreScope(const StmtPtr& stmt);

// Wrap statements that lack InCore coverage in ScopeStmt(InCore).
// After InterchangeChunkLoops processes the auto_incore body, some statements
// may lack InCore wrapping. This function groups consecutive such statements
// and wraps each group in ScopeStmt(InCore).
static StmtPtr WrapNonIncoreStatementsInInCore(const StmtPtr& body, const Span& span);
```

**Transformation:**
- Ensures loop order: outer orchestration loops → inner parallel loops
- Wraps non-InCore statements in `ScopeStmt(InCore)`

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | Minor reordering and index expression adjustments |

---

## Pass 06: RunVerifier

**Source File:** `src/ir/verifier/*.cpp`

**Purpose:** Validates IR correctness including SSA properties, type consistency, and structural requirements.

**Key Verifications:**
- `verify_ssa_pass.cpp` - Ensures SSA form (single definition per variable)
- `type_check_pass.cpp` - Type consistency checks
- `verify_no_nested_call_pass.cpp` - No nested call expressions remain

| PA4 Status | **NO CHANGE** |
|------------|---------------|
| Reason | Verification pass; no IR transformation, only validation |

---

## Pass 07: OutlineIncoreScopes ⭐⭐

**Source File:** `src/ir/transforms/outline_incore_scopes_pass.cpp`

**Purpose:** Extracts `ScopeStmt(InCore)` bodies into separate `Function(InCore)` definitions, converting the parent function to `FunctionType::Orchestration`.

**Key Logic from Source:**
```cpp
// Pass to outline InCore scopes into separate functions
//
// Transformation:
// 1. For each ScopeStmt(InCore) in an Opaque function:
//    - Analyze body to determine external variable references (inputs)
//    - Analyze subsequent statements to determine which definitions are outputs
//    - Extract body into new Function(InCore) with appropriate params/returns
//    - Replace scope with Call to the outlined function + output assignments
// 2. Recursively handles nested InCore scopes
// 3. Add outlined functions to the program
// 4. Promote the parent function from Opaque to Orchestration
```

**Transformation:**

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | Single function split into `paged_attention` (Orchestration) + `paged_attention_incore_0` (InCore) |

**Before (single function):**
```python
def paged_attention(...):
    with pl.auto_incore():
        for b_idx_0_out in pl.range(...):
            for b_idx_0_in in pl.parallel(...):
                # ... compute ...
```

**After (two functions):**
```python
@pl.function(type=pl.FunctionType.Orchestration)
def paged_attention(...):
    for b_idx_0_out in pl.range(0, 8, 1, ...):
        for q_idx_0_out in pl.range(0, 2, 1, ...):
            result = self.paged_attention_incore_0(b_idx_0_out, ..., q_idx_0_out, ...)

@pl.function(type=pl.FunctionType.InCore)
def paged_attention_incore_0(...):
    for b_idx_0_in in pl.parallel(0, 8, 1, ...):
        for q_idx_0_in in pl.parallel(0, 2, 1, ...):
            # ... actual compute on AI Core ...
```

---

## Pass 08: ExpandMixedKernel ⭐⭐⭐

**Source File:** `src/ir/transforms/expand_mixed_kernel_pass.cpp`

**Purpose:** Splits InCore kernels containing both Cube (MatMul) and Vector operations into separate AIC (AI Core Cube) and AIV (AI Vector) functions, inserting TPUSH/TPOP communication primitives.

**Key Logic from Source:**
```cpp
// Core Affinity Classification
enum class CoreAffinity { CUBE, VECTOR, SHARED, MIXED };

bool IsCubeOp(const std::string& name) {
    static const std::unordered_set<std::string> cube_ops = {
        "tile.matmul", "tile.matmul_acc", "tile.matmul_bias", 
        "tile.gemv", "tile.gemv_acc", "tile.gemv_bias", "tile.batch_matmul"
    };
    return cube_ops.count(name) > 0;
}

CoreAffinity ClassifyCallAffinity(const CallPtr& call) {
    // Cube ops → CUBE
    if (IsCubeOp(name)) return CoreAffinity::CUBE;
    // tile.* ops that are not cube → VECTOR
    if (name.substr(0, 5) == "tile.") return CoreAffinity::VECTOR;
    return CoreAffinity::SHARED;
}
```

**Transformation:**

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | `paged_attention_incore_0` split into `_aic` (Cube) + `_aiv` (Vector) + `function_group` |

**Key Changes:**

1. **AIC Function (Cube Unit):**
   - Only MatMul operations
   - Receives data via `pl.comm.tpop_from_aiv(slot)`
   - Sends results via `pl.comm.tpush_to_aiv(tile, slot)`
   - Tile assembly: combines two 8×128 halves into 16×128

2. **AIV Function (Vector Unit):**
   - All vector operations (mul, exp, row_max, row_sum, div, etc.)
   - Loads from DDR
   - Sends to AIC via `pl.comm.tpush_to_aic(tile, AIV_IDX)`
   - Receives from AIC via `pl.comm.tpop_from_aic(AIV_IDX)`
   - Has `AIV_IDX` runtime parameter for multi-AIV parallelism

3. **Function Group:**
   ```python
   @pl.function_group(
       aic="paged_attention_incore_0_aic",
       aiv="paged_attention_incore_0_aiv",
       aiv_runtime_params=["AIV_IDX"]
   )
   class paged_attention_incore_0_group:
       pass
   ```

**Data Flow Diagram:**
```
┌─────────────────────────────────────────────────────────────────────┐
│                        AIV (Vector Unit)                             │
│  DDR → qi[8,128] ─────────────────────────────────────────────────┐ │
│  DDR → kj[64,128] ────────────────────────────────────────────────┼─│─┐
│  DDR → vj[64,128] ────────────────────────────────────────────────┼─│─┼─┐
└───────────────────────────────────────────────────────────────────┼─┼─┼─┘
                                                                     │ │ │
                         TPUSH_TO_AIC (slot 0,1)                     │ │ │
                                                                     ▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AIC (Cube Unit)                               │
│  qi[16,128] ← assemble(tpop[0], tpop[1])                            │
│  kj[128,128] ← assemble(tpop[0], tpop[1])                           │
│  vj[128,128] ← assemble(tpop[0], tpop[1])                           │
│                                                                      │
│  sij[16,128] = matmul(qi, kj^T)  ───────────────────────────────────┼─┐
│  oi_tmp[16,128] = matmul(pij, vj) ──────────────────────────────────┼─┼─┐
└─────────────────────────────────────────────────────────────────────┼─┼─┘
                                                                       │ │
                         TPUSH_TO_AIV (slot 0,1)                       │ │
                                                                       ▼ ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        AIV (Vector Unit)                             │
│  sij[8,128] ← tpop_from_aic(AIV_IDX)                                │
│                                                                      │
│  scaled = mul(sij, scale)                                           │
│  mi = row_max(scaled)                                               │
│  sij_centered = sub(scaled, mi)                                     │
│  exp_vals = exp(sij_centered)                                       │
│  pij = cast(exp_vals, BF16)                                         │
│  li = row_sum(pij)                                                  │
│  ... online softmax update ...                                      │
│  out = div(oi_updated, li_new)                                      │
│                                                                      │
│  assemble(out, dst) → DDR                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pass 09: ConvertTensorToBlockOps

**Source File:** `src/ir/transforms/convert_tensor_to_tile_ops_pass.cpp`

**Purpose:** Converts high-level `tensor.*` operations to low-level `tile.*` or `block.*` operations with explicit memory management.

**Key Logic from Source:**
```cpp
// Visitor that collects tensor-typed variable names used directly by converted ops.
// Traverses the IR tree and records the name of every Var/IterArg argument
// whose type is TensorType and that appears in a call to an op registered in
// OpConversionRegistry (i.e. an op that will be converted from tensor.* to tile.*).

// Return value memory space rules for tile operators
const std::map<std::string, std::optional<MemorySpace>> kTileOpMemoryRules = {
    {"tile.create", std::nullopt},          // Extract from target_memory
    {"tile.load", std::nullopt},            // Extract from target_memory
    {"tile.move", std::nullopt},            // Extract from target_memory
    {"tile.store", MemorySpace::DDR},       // Fixed DDR
    {"tile.matmul", MemorySpace::Acc},      // Fixed Acc
    {"tile.matmul_acc", MemorySpace::Acc},  // Fixed Acc
};
```

**Typical Transformation (for other kernels):**
```python
# Before
result = pl.tensor.matmul(a, b)

# After  
tile_a = pl.tile.load(a, offsets, shapes, target=Mat)
tile_b = pl.tile.load(b, offsets, shapes, target=Mat)
tile_a_l0 = pl.tile.move(tile_a, target=Left)
tile_b_l0 = pl.tile.move(tile_b, target=Right)
tile_c = pl.tile.matmul(tile_a_l0, tile_b_l0)
pl.tile.store(tile_c, result, offsets)
```

| PA4 Status | **MINIMAL CHANGE** |
|------------|---------------------|
| Reason | Most tensor ops already converted during ExpandMixedKernel; TPUSH/TPOP handle data movement |

---

## Pass 10: InitMemRef ⭐

**Source File:** `src/ir/transforms/init_memref.cpp`

**Purpose:** Annotates all tensor/tile variables with `MemRef` descriptors specifying memory space, size, and buffer ID.

**Key Logic from Source:**
```cpp
// Helper to extract target_memory from Call kwargs
MemorySpace ExtractTargetMemory(const CallPtr& call);

// Return value memory space rules for tile operators
const std::map<std::string, std::optional<MemorySpace>> kTileOpMemoryRules = {
    {"tile.create", std::nullopt},          // Extract from target_memory
    {"tile.load", std::nullopt},            // Extract from target_memory
    {"tile.store", MemorySpace::DDR},       // Fixed DDR
    {"tile.matmul", MemorySpace::Acc},      // Fixed Acc
};

// Visitor to identify memory space for each variable
class MemRefUsageVisitor : public IRVisitor {
    // Initialize with function parameters (all params should be in DDR)
    explicit MemRefUsageVisitor(const std::vector<VarPtr>& params, ...) {
        for (const auto& var : params) {
            var_memory_spaces_[var] = MemorySpace::DDR;
        }
    }
};
```

**Transformation:**

| PA4 Status | **CHANGED** ✅ |
|------------|----------------|
| Changes | All tensors annotated with MemRef(space, offset, size, buffer_id) |

**Before:**
```python
query_0: pl.Tensor[[4096, 128], pl.BFLOAT16]
qi_0: pl.Tensor[[16, 128], pl.BFLOAT16]
```

**After:**
```python
query_0: pl.Tensor[[4096, 128], pl.BFLOAT16, pl.MemRef(
    pl.MemorySpace.DDR,  # Memory space
    -1,                   # Offset (unresolved)
    1048576,              # Size: 4096×128×2 bytes
    0                     # Buffer ID
)]

qi_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(
    pl.MemorySpace.DDR,
    -1,                   # Offset (unresolved)
    4096,                 # Size: 16×128×2 bytes
    12                    # Buffer ID
)]
```

---

## Pass 11: MemoryReuse

**Source File:** `src/ir/transforms/basic_memory_reuse_pass.cpp`

**Purpose:** Analyzes buffer lifetimes and identifies opportunities for memory reuse to reduce on-chip memory footprint.

**Key Logic from Source:**
```cpp
// Lifetime interval for a TileType variable (based on topological order)
struct LifetimeInterval {
    VarPtr variable;           // The variable
    int def_point;             // Definition point (topological order)
    int last_use_point;        // Last use point (topological order)
    MemorySpace memory_space;  // Memory space
    uint64_t size;             // Size in bytes
    bool no_reuse = false;     // If true, must NOT reuse another variable's memory
};

// Assign topological order to all statements in basic blocks
std::map<StmtPtr, int> AssignDeclarationOrder(const std::vector<BasicBlock>& blocks);
```

**Transformation:**
- Computes def-use chains for all tile variables
- Identifies non-overlapping lifetimes
- Groups variables that can share memory allocations

| PA4 Status | **NO VISIBLE CHANGE** |
|------------|------------------------|
| Reason | Analysis pass; results used by AllocateMemoryAddr; no IR text difference |

---

## Pass 12: InsertSync

**Source File:** `src/ir/transforms/insert_sync_pass.cpp`

**Purpose:** Inserts pipeline synchronization primitives (`set_flag`/`wait_flag`) based on data dependencies between different hardware pipelines.

**Key Logic from Source:**
```cpp
// Path element representing a position in the IR tree
struct PathElement {
    enum class Kind { SeqIndex, OpIndex, IfThen, IfElse, ForBody };
    Kind kind;
    int index;
};

// Position in the IR tree, represented as a path from root
struct Position {
    std::vector<PathElement> path;
    
    // Determines if this position and another are within the same control flow scope
    bool IsInSameScope(const Position& other) const;
    
    // Determines if this position is before another
    bool IsBefore(const Position& other) const;
};
```

**Sync Types:**
- `PIPE_MTE2` → `PIPE_MTE1`: GM → L1 complete before L1 → L0
- `PIPE_MTE1` → `PIPE_M`: L1 → L0 complete before Cube compute
- `PIPE_M` → `PIPE_MTE3`: Cube complete before L0C → GM store
- `PIPE_V` → `PIPE_MTE3`: Vector complete before UB → GM store

| PA4 Status | **NO VISIBLE CHANGE** |
|------------|------------------------|
| Reason | Sync primitives already present in ExpandMixedKernel output (TPUSH/TPOP imply sync); additional fine-grained sync may be added but not visible in Python IR |

---

## Pass 13: AllocateMemoryAddr

**Source File:** `src/ir/transforms/allocate_memory_addr_pass.cpp`

**Purpose:** Resolves MemRef offset values (`-1` → concrete addresses) and performs final memory layout.

**Key Logic from Source:**
```cpp
// Helper function to align address to 32-byte boundary
inline uint64_t Align32(uint64_t addr) { return (addr + 31) & ~31ULL; }

// Visitor to collect all MemRef objects from TileType variables
class MemRefCollectorVisitor : public IRVisitor {
    void VisitExpr_(const VarPtr& op) override {
        auto tile_type = std::dynamic_pointer_cast<const TileType>(op->GetType());
        if (tile_type && tile_type->memref_.has_value()) {
            AddMemRefIfUnique(tile_type->memref_.value());
        }
    }
};
```

**Transformation:**
- Collects all MemRef objects
- Groups by memory space (DDR, UB, L0A, L0B, L0C)
- Assigns concrete offsets with 32-byte alignment
- Applies memory reuse decisions from Pass 11

| PA4 Status | **NO VISIBLE CHANGE** |
|------------|------------------------|
| Reason | Offset resolution happens in backend codegen; Python IR shows `offset=-1` throughout |

---

## Architecture Mapping

```
┌──────────────────────────────────────────────────────────────────┐
│                     Ascend NPU (Da Vinci)                        │
├──────────────────────────────────────────────────────────────────┤
│  Orchestration (ARM CPU)                                         │
│    └── pl.range loops, kernel dispatch                           │
├──────────────────────────────────────────────────────────────────┤
│  AI Core (×N)                                                    │
│    ├── AIC (Cube Unit) ──── MatMul, Conv                         │
│    │     └── L0A, L0B, L0C buffers                               │
│    │                                                             │
│    ├── AIV (Vector Unit) ── Element-wise, Reduce, Softmax        │
│    │     └── UB (Unified Buffer)                                 │
│    │                                                             │
│    └── TPUSH/TPOP ───────── Ring buffer AIC↔AIV (8 slots)        │
├──────────────────────────────────────────────────────────────────┤
│  Global Memory (DDR/HBM)                                         │
│    └── Tensors: query, key_cache, value_cache, out               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Summary: Pass Change Matrix for PA4

| Pass | Source File | Changes in PA4? | Key Transformation |
|------|-------------|-----------------|-------------------|
| 01 UnrollLoops | `unroll_loops_pass.cpp` | ❌ No | Compile-time loop expansion |
| 02 ConvertToSSA | `convert_to_ssa_pass.cpp` | ✅ Yes | SSA + init_values + yield |
| 03 FlattenCallExpr | `flatten_call_expr_pass.cpp` | ❌ No | Three-address code |
| 04 SplitChunkedLoops | `split_chunked_loops_pass.cpp` | ✅ Yes | Outer/inner loop split |
| 05 InterchangeChunkLoops | `interchange_chunk_loops_pass.cpp` | ✅ Yes | Loop reordering |
| 06 RunVerifier | `verifier/*.cpp` | ❌ No | Validation only |
| 07 OutlineIncoreScopes | `outline_incore_scopes_pass.cpp` | ✅ Yes | Function separation |
| 08 ExpandMixedKernel | `expand_mixed_kernel_pass.cpp` | ✅ Yes | AIC/AIV split + TPUSH/TPOP |
| 09 ConvertTensorToBlockOps | `convert_tensor_to_tile_ops_pass.cpp` | ⚠️ Minimal | Tensor → Tile ops |
| 10 InitMemRef | `init_memref.cpp` | ✅ Yes | MemRef annotations |
| 11 MemoryReuse | `basic_memory_reuse_pass.cpp` | ❌ No (internal) | Lifetime analysis |
| 12 InsertSync | `insert_sync_pass.cpp` | ❌ No (internal) | Pipeline sync |
| 13 AllocateMemoryAddr | `allocate_memory_addr_pass.cpp` | ❌ No (internal) | Address allocation |

**Legend:**
- ✅ Yes: Visible IR changes in PA4 dump
- ❌ No: No visible changes (either pass doesn't apply or changes are internal)
- ⚠️ Minimal: Minor or no visible changes due to prior passes
