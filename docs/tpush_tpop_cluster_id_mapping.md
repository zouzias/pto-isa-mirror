# Appendix: Cluster ID Mapping and Core Architecture Assumptions

## Overview

This appendix describes the **cluster ID (CVID) mapping** assumptions for A5 and A2A3 platforms, which underpins the TPUSH/TPOP ring buffer communication design.

## Platform Architecture Comparison

| Aspect | A5 | A2A3 |
|--------|-----|------|
| **Architecture** | Tightly coupled | Decoupled |
| **Cluster Binding** | Hardware-fixed 1:2 mapping | Task scheduler-bound |
| **CVID Source** | Direct core ID computation | GM slot + cross-core sync |
| **Sync Mechanism** | SET intra-block | SET cross-core via FFTS |
| **Local Datapath** | L0C↔UB, UB↔L1 direct | Via GM staging |

## A5: Direct Cluster Mapping

### Architecture

On A5, each die contains **18 core clusters** with a fixed 1:2 architecture:

```
Die Layout:
  CORE_PER_DIE = 18
  AIV_RATIO = 2

    +--------------------------------------------------------------+
    |  Die                                                         |
    |  +----------+----------+----------+-------------------------+
    |  |Cluster 0 |Cluster 1 |Cluster 2 | ...  Cluster 17         |
    |  | AIC+2xAIV| AIC+2xAIV| AIC+2xAIV|      AIC+2xAIV          |
    |  +----------+----------+----------+-------------------------+
    |                                                              |
    |  Total cores per die: 18 x (1 + 2) = 54                      |
    +--------------------------------------------------------------+
```

### CVID Computation

The cluster ID is computed **directly from the physical core ID** without runtime negotiation:

```cpp
// A5 TSYNC_CVID implementation
#ifdef __DAV_CUBE__
    int die_id = get_coreid() / AIC_AIV_PER_DIE;     // AIC_AIV_PER_DIE = 54
    comm_slot = die_id * CORE_PER_DIE + get_coreid() % AIC_AIV_PER_DIE;
#elif defined(__DAV_VEC__)
    int die_id = get_coreid() / AIC_AIV_PER_DIE;
    comm_slot = die_id * CORE_PER_DIE + 
                (((get_coreid() % AIC_AIV_PER_DIE) - CORE_PER_DIE - get_subblockid()) / AIV_RATIO);
#endif
```

**Key properties**:
- **No runtime communication** needed to establish cluster ID
- **Deterministic mapping**: core ID → cluster ID is a pure function
- **Hardware-enforced 1:2 relationship**: L0C↔UB and UB↔L1 local datapaths exist within each cluster
- **Intra-block sync**: `SET flag` for synchronization within the cluster (no GM round-trip)

### Cluster Structure

```
Cluster cvid=N (A5):

    +---------------------------------------------------------+
    |                    Local Datapaths                      |
    |                                                         |
    |  +----------+    L0C<->UB    +----------+               |
    |  |  AIV 0   |<------------->|          |               |
    |  |  (UB)    |               |   AIC    |               |
    |  +----------+               |  (L0C)   |               |
    |                             |          |               |
    |  +----------+    UB<->L1    |          |               |
    |  |  AIV 1   |<------------->|          |               |
    |  |  (UB)    |               +----------+               |
    |  +----------+                                          |
    |                                                         |
    |  Sync: SET intra-block (hardware flags, 8 per dir)      |
    +---------------------------------------------------------+
```

## A2A3: Decoupled Task-Scheduled Mapping

### Architecture

On A2A3, the Cube and Vector cores are **physically decoupled**. The 1:2 cluster relationship is established **at runtime by the hardware task scheduler**, not by physical topology.

### CVID Negotiation

The cluster ID is communicated **through GM** using the FFTS cross-core sync mechanism:

```cpp
// A2A3 TSYNC_CVID implementation
#ifdef __DAV_CUBE__
    // Cube core writes its core ID to GM slot
    comm_slot = static_cast<int>(get_coreid() & 0x7f);
    comm_slot %= CV_MAX_CORES;
    
    // Write to GM slot and flush cache
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    comm_slot_ptr[0] = static_cast<uint32_t>(comm_slot);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    dsb(DSB_DDR);
    
    // Signal Vector cores via FFTS
    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, CV_COMM_CTRL));
    
#elif defined(__DAV_VEC__)
    // Vector core waits for Cube's signal, then reads cluster ID from GM
    __gm__ volatile uint32_t *comm_slot_ptr = reinterpret_cast<__gm__ volatile uint32_t *>(
        cv_comm_buf + static_cast<std::size_t>(block_idx) * CV_COMM_SLOT_BYTES);
    dcci(comm_slot_ptr, SINGLE_CACHE_LINE);
    wait_flag_dev(CV_COMM_CTRL);
    comm_slot = static_cast<int>(comm_slot_ptr[0]);
#endif
```

**Key properties**:
- **Runtime negotiation**: Cube core announces its ID via GM; Vector cores read it after FFTS sync
- **Task scheduler binding**: The `block_idx` ↔ `core_id` mapping is established by AICPU when launching the task
- **FFTS cross-core sync**: Uses `ffts_cross_core_sync()` and `wait_flag_dev()` for signaling
- **GM-based communication**: Cluster ID travels through GM (`cv_comm_buf`)

### Cluster Binding Flow

```
A2A3 Task Launch (via AICPU):

    +---------------------------------------------------------------------+
    |  AICPU (Task Scheduler)                                             |
    |                                                                     |
    |  1. Allocates block_idx for each task                               |
    |  2. Assigns physical Cube core (core_id)                            |
    |  3. Assigns 2 buddy Vector cores (subblock 0, 1)                    |
    |  4. Creates logical cluster: block_idx <-> {AIC, AIV0, AIV1}        |
    +---------------------------------------------------------------------+
                |
                v
    +---------------------------------------------------------------------+
    |  Runtime: Cluster ID Exchange via GM                                |
    |                                                                     |
    |   AIC writes:  cv_comm_buf[block_idx * 512] = core_id & 0x7f        |
    |                dcci + dsb                                           |
    |                ffts_cross_core_sync(CV_CORE_SYNC)                   |
    |                                                                     |
    |   AIV waits:   wait_flag_dev(CV_COMM_CTRL)                          |
    |   AIV reads:   comm_slot = cv_comm_buf[block_idx * 512]             |
    +---------------------------------------------------------------------+
                |
                v
    +---------------------------------------------------------------------+
    |  Result: All cores in cluster share same comm_slot (CVID)           |
    +---------------------------------------------------------------------+
```

## PyPTO Runtime Assumption

The TPUSH/TPOP ISA design **assumes the PyPTO runtime follows the same cluster architecture** when launching mixed InCore tasks:

1. **AICPU dispatches** the mixed function group (Cube kernel + Vector kernel) with a consistent `block_idx`.

2. **Cluster ID is resolved**:
   - **A5**: Direct computation from `get_coreid()` — no runtime setup needed
   - **A2A3**: Runtime `TSYNC_CVID()` call with GM slot exchange

3. **GM_SLOT_BUFFER indexing**: After CVID is resolved, the kernel computes:
   ```
   my_gm_slot_buffer = GM_SLOT_BUFFER_BASE + CVID * PER_CLUSTER_SLOT_BUFFER_SIZE
   ```

4. **TPUSH/TPOP flow control**: Uses the hardware flags (8 per direction) for P2C/C2P signaling within the resolved cluster.

### Invariant

**All cores in the same cluster must resolve to the same CVID.**  
This is guaranteed by:
- **A5**: Hardware topology (core IDs within a cluster compute the same CVID)
- **A2A3**: AICPU dispatch + GM negotiation via FFTS sync

## A2A3: Working Buffer Reservation for CVID Communication

On A2A3, the CVID negotiation requires a **reserved region at the bottom of the working buffer** (GM) for the `cv_comm_buf` slots used during `TSYNC_CVID()`.

### Reserved Space Calculation

```
CV_COMM_SLOT_BYTES = 512 bytes (per block, 512B aligned)
CV_MAX_CORES       = 25 (max block_dim)

Reserved space = CV_COMM_SLOT_BYTES * CV_MAX_CORES
               = 512 * 25
               = 12,800 bytes
               = 12.5 KB (round up to 16KB for alignment)
```

**Practical allocation**: Reserve **12.5KB** (512B × 25 blocks) at the bottom of the working buffer for CVID communication initialization. Round up to **16KB** for alignment safety. The runtime must ensure this region is not used for other purposes.

### Memory Layout

```
A2A3 Working Buffer (GM):

    +------------------------------------------------------------------+
    |  Bottom 12.5KB: Reserved for cv_comm_buf (CVID negotiation)      |
    |                                                                  |
    |  +------------+------------+------------+-----+------------+     |
    |  | block_idx=0| block_idx=1| block_idx=2| ... | block_idx=24|    |
    |  |   512B     |   512B     |   512B     |     |   512B      |    |
    |  +------------+------------+------------+-----+------------+     |
    |                                                                  |
    +------------------------------------------------------------------+
    |  Remaining space: Available for GM_SLOT_BUFFER, task data, etc.  |
    +------------------------------------------------------------------+
```

### Why This Reservation?

1. **CVID exchange at kernel startup**: Before TPUSH/TPOP can operate, Cube and Vector cores must agree on the cluster ID. The Cube writes its `core_id & 0x7f` to `cv_comm_buf[block_idx * 512]` and signals via FFTS; Vector cores read after sync.

2. **Cannot overlap with GM_SLOT_BUFFER**: The `cv_comm_buf` is used **before** the ring buffer is initialized. If it overlapped with `GM_SLOT_BUFFER`, the CVID write could corrupt pending TPUSH data (or vice versa).

3. **Fixed at runtime init**: The PyPTO runtime (simpler/L0–L2) must reserve this space unconditionally on A2A3 when mixed InCore tasks are enabled.

### A5: No Working Buffer Reservation Needed

On A5, the CVID is computed directly from `get_coreid()` without any GM communication. No working buffer reservation is required for cluster ID negotiation.

## Constants Reference

| Constant | A5 Value | A2A3 Value | Description |
|----------|----------|------------|-------------|
| `CORE_PER_DIE` | 18 | 25 | Clusters per die |
| `AIV_RATIO` | 2 | 2 | Vector cores per Cube |
| `AIC_AIV_PER_DIE` | 54 | 75 | Total cores per die (AIC + AIV) |
| `CV_COMM_SLOT_BYTES` | 512 | 512 | Bytes per block's comm slot |
| `CV_MAX_CORES` | 36 | 25 | Max clusters supported |
| `CV_COMM_RESERVED` | 0 | 12.5KB | Bottom working buffer reservation for CVID |

## Related Documents

- [Source: A5 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a5/custom/TSyncCVID.hpp)
- [Source: A2A3 TSyncCVID.hpp](https://gitcode.com/cann/pto-isa/blob/master/include/pto/npu/a2a3/custom/TSyncCVID.hpp)



## A3 FFTS Scheduler and Logical Cluster Setup

### Current TPUSH/TPOP Implementation on A3

The current A3 TPUSH/TPOP implementation relies on the **FFTS (Fast Function Task Scheduler) cross-core synchronization** features. During mixed kernel kickstart, the FFTS hardware establishes a **logical cluster** through the block scheduler.

### Logical Cluster Formation

When FFTS launches a mixed kernel, it creates a logical cluster with a **1:2 block-subblock relationship**:

```
FFTS Mixed Kernel Kickstart:

    +---------------------------------------------------------------------+
    |  FFTS Block Scheduler                                               |
    |                                                                     |
    |  Kickstart: 1 block + 2 subblocks (1:2 ratio)                       |
    |                                                                     |
    |  +-------------------+                                              |
    |  | Logical Cluster   |                                              |
    |  |                   |                                              |
    |  |  Block (AIC)      |  <-- block_id identifies the cluster         |
    |  |    |              |                                              |
    |  |    +-- Subblock 0 |  <-- subblock_id = 0 (AIV0)                  |
    |  |    +-- Subblock 1 |  <-- subblock_id = 1 (AIV1)                  |
    |  |                   |                                              |
    |  +-------------------+                                              |
    |                                                                     |
    +---------------------------------------------------------------------+
```

### Logical-to-Physical Core Mapping

The FFTS hardware builds the **logical-to-physical core mapping** at task launch time:

1. **Block ID → Physical Cube Core**: The block scheduler assigns a physical AIC core to each logical block.

2. **Subblock ID → Physical Vector Core**: Each subblock (0, 1) maps to a physical AIV core that becomes a buddy of the assigned Cube.

3. **Intra-cluster sync resolution**: The FFTS hardware resolves all intra-cluster synchronization and communication paths based on this mapping.

### Kernel Identification of Cluster

Kernels running on Cube and Vector cores can identify their **logical block cluster** using:

- **Cube core**: `get_block_idx()` — returns the logical block ID
- **Vector core**: `get_block_idx()` + `get_subblockid()` — block ID identifies the cluster, subblock ID identifies AIV0 vs AIV1

```cpp
// On Cube (AIC)
int my_cluster = get_block_idx();

// On Vector (AIV)
int my_cluster = get_block_idx();
int my_aiv_idx = get_subblockid();  // 0 or 1
```

### PyPTO Recommendation: Logical Block ID as Cluster ID

When PyPTO initializes a task with **`block_dim == num_cores`** (number of physical core clusters), we recommend using the **logical block ID directly as the cluster ID** for GM FIFO access:

```cpp
// Recommended: use logical block_idx as cluster_id
// (when block_dim == num_cores, no wastage)
int cluster_id = get_block_idx();

// GM_SLOT_BUFFER access
my_gm_slot_buffer = GM_SLOT_BUFFER_BASE + cluster_id * PER_CLUSTER_SLOT_BUFFER_SIZE;
```

**Rationale**: Since there is a 1:1 mapping between logical blocks and physical clusters (no over-subscription), using `block_idx` directly avoids the overhead of GM-based CVID negotiation while maintaining correctness.

### AICPU Runtime Requirement

When AICPU kicks off a runtime on a cluster, it **must follow the same logical block_idx → cluster mapping** as the FFTS-launched kernels:

1. **Consistent block_idx assignment**: AICPU must assign `block_idx` values that match the FFTS block scheduler's logical cluster numbering.

2. **Physical core binding**: The physical Cube and Vector cores assigned by AICPU must form the same 1:2 cluster relationship.

3. **Synchronization compatibility**: The AICPU-launched runtime must be able to use the same FFTS cross-core sync primitives (`ffts_cross_core_sync`, `wait_flag_dev`) as the kernel code.

```
AICPU + FFTS Coordination:

    AICPU                           FFTS Block Scheduler
    +---------------------------+   +---------------------------+
    | Launch runtime on         |   | Launch kernel on          |
    | cluster with block_idx=N  |   | cluster with block_idx=N  |
    |                           |   |                           |
    | Physical cores:           |   | Physical cores:           |
    |   AIC = core_X            |   |   AIC = core_X            |
    |   AIV0 = core_Y           |   |   AIV0 = core_Y           |
    |   AIV1 = core_Z           |   |   AIV1 = core_Z           |
    +---------------------------+   +---------------------------+
                |                               |
                +-------------------------------+
                              |
                              v
                    Same logical cluster mapping
```

This ensures that TPUSH/TPOP ring buffer operations work correctly whether the kernel is launched via FFTS directly or through AICPU runtime initialization.
## Alternative: Logical Block ID as Cluster ID

If PyPTO **always launches with `block_dim` equal to the number of physical core clusters**, a simpler mapping is possible:

```
cluster_id = block_idx
```

### Conditions for This Simplification

1. **block_dim == number of physical clusters**: The runtime guarantees a 1:1 mapping between logical blocks and physical clusters.

2. **No over-subscription**: Each cluster runs exactly one block — no time-sharing of clusters across multiple blocks.

3. **Consistent dispatch**: AICPU assigns `block_idx=0` to cluster 0, `block_idx=1` to cluster 1, etc.

### Advantages

- **No GM communication needed on A2A3**: Skip the `TSYNC_CVID()` handshake entirely
- **No working buffer reservation**: The 12.5KB `cv_comm_buf` region is not required
- **Simpler kernel startup**: Both Cube and Vector cores use `block_idx` directly as the cluster ID

### Implementation

```cpp
// Simplified cluster ID resolution (when block_dim == num_clusters)
inline int get_cluster_id() {
    return get_block_idx();  // Direct mapping, no GM exchange
}

// GM_SLOT_BUFFER indexing
my_gm_slot_buffer = GM_SLOT_BUFFER_BASE + get_block_idx() * PER_CLUSTER_SLOT_BUFFER_SIZE
```

### Trade-off

This approach **sacrifices flexibility** (cannot run fewer blocks than clusters, or share clusters) for **simplicity** (no runtime CVID negotiation). If PyPTO's execution model always uses full cluster utilization, this is the preferred approach.
