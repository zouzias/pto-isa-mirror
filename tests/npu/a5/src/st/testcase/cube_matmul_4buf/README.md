# cube_matmul_4buf Test Case

4-buffer matmul kernel demonstrating software pipelining with explicit synchronization.

## Test Configuration

- **Dimensions:** M=32, K=16, N=256, K_ITERS=64
- **Total K:** 1024 (64 iterations × 16 per tile)
- **Data types:** Input fp16, Output fp32

## Buffer Layout

### A5 Hardware Capacity
| Buffer | Capacity |
|--------|----------|
| L1 (Mat) | 512 KB |
| L0A (Left) | 64 KB |
| L0B (Right) | 64 KB |
| L0C (Acc) | 256 KB |

### Our Allocation
| Buffer | A Tiles | B Tiles | Total Used |
|--------|---------|---------|------------|
| L1 | 4 × 1KB = 4KB | 4 × 8KB = 32KB | ~36KB (fits 512KB) ✅ |
| L0A | 4 × 1KB = 4KB | - | 4KB (fits 64KB) ✅ |
| L0B | - | 4 × 8KB = 32KB | 32KB (fits 64KB) ✅ |
| L0C | - | - | 32KB (fits 256KB) ✅ |

### L1 Address Map
| Buffer | A Tiles | B Tiles |
|--------|---------|---------|
| 0 | 0x0000 | 0x10000 |
| 1 | 0x0800 | 0x12000 |
| 2 | 0x1000 | 0x14000 |
| 3 | 0x1800 | 0x16000 |

### L0 Address Map
| Buffer | L0A (A) | L0B (B) |
|--------|---------|---------|
| 0 | 0x000 | 0x0000 |
| 1 | 0x400 | 0x2000 |
| 2 | 0x800 | 0x4000 |
| 3 | 0xC00 | 0x6000 |

## Synchronization Methods Comparison

### Test Results (A5 PEM Simulator)

| Sync Method | Cycles | Max Diff | Status |
|-------------|--------|----------|--------|
| set_flag/wait_flag | 24,276 | 9.54e-06 | ✅ PASSED |
| pipe_barrier(PIPE_ALL) | 52,957 | 9.54e-06 | ✅ PASSED |
| get_buf/rls_buf | 23,803 | 12.6 | ❌ FAILED |

### 1. Event-ID Based (set_flag/wait_flag) - RECOMMENDED
```cpp
// Syntax
set_flag(src_pipe, dst_pipe, event_id);
wait_flag(src_pipe, dst_pipe, event_id);

// Example: MTE2 → MTE1 sync
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
```

**Pros:** Reliable, well-tested, explicit cross-pipeline sync
**Cons:** Requires understanding of pipeline pairs

### 2. Barrier-Based (pipe_barrier) - SAFE BUT SLOW
```cpp
pipe_barrier(PIPE_ALL);  // Waits for ALL pipelines
```

**Pros:** Simple, guaranteed correct
**Cons:** ~2x slower (serializes all pipelines)

### 3. Buffer-ID Based (get_buf/rls_buf) - NOT WORKING

CCE built-in functions cannot be called with template syntax. Wrapper required:

```cpp
// CCE built-in signature (non-template)
get_buf(pipe, buf_id, mode);
rls_buf(pipe, buf_id, mode);

// Wrapper template for cleaner syntax
template <int pipe>
AICORE void get_buffer(int id) {
    get_buf(pipe, id, 0);  // mode=0
}

template <int pipe>
AICORE void rls_buffer(int id) {
    rls_buf(pipe, id, 0);
}
```

**Status:** ❌ FAILED - produces wrong results
**Investigation needed:**
- Correct `mode` parameter values
- Cross-pipeline synchronization semantics
- May require additional sync between stages

## Build & Run

```bash
# Build
source /usr/local/Ascend/cann_9b2/cann/set_env.sh
cd ~/pto-isa-ptoas-st
python3 tests/script/build_st.py -r sim -v a5 -t cube_matmul_4buf

# Run
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
cd tests/npu/a5/src/st/build
python3 ../testcase/cube_matmul_4buf/gen_data.py
./bin/cube_matmul_4buf
```

## References

- ptoas-insert-sync-analysis.md Section 10 (Buffer-ID approach)
- CCE intrinsics: `__builtin_cce_get_buf`, `__builtin_cce_rls_buf`
- A5 architecture: 64KB L0A, 64KB L0B, 256KB L0C, 512KB L1
