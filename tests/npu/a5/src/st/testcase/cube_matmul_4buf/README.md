# cube_matmul_4buf Test Case

4-buffer matmul kernel demonstrating software pipelining with explicit synchronization.

## Test Configuration

- **Dimensions:** M=32, K=16, N=256, K_ITERS=64
- **Total K:** 1024 (64 iterations × 16 per tile)
- **Data types:** Input fp16, Output fp32

## Buffer Layout

### L1 Buffers
| Buffer | A Tiles | B Tiles |
|--------|---------|---------|
| 0 | 0x0000 | 0x10000 |
| 1 | 0x0800 | 0x12000 |
| 2 | 0x1000 | 0x14000 |
| 3 | 0x1800 | 0x16000 |

### L0 Buffers
| Buffer | L0A (A) | L0B (B) |
|--------|---------|---------|
| 0 | 0x000 | 0x0000 |
| 1 | 0x400 | 0x2000 |
| 2 | 0x800 | 0x4000 |
| 3 | 0xC00 | 0x6000 |

## Synchronization Methods

### 1. Event-ID Based (set_flag/wait_flag)
```cpp
// Syntax
set_flag(src_pipe, dst_pipe, event_id);
wait_flag(src_pipe, dst_pipe, event_id);

// Example
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
```

**Pros:** Well-tested, reliable
**Cons:** Requires pre-loop priming and post-loop draining for multi-buffer patterns

### 2. Buffer-ID Based (get_buf/rls_buf) - CCE Intrinsics

CCE built-in functions cannot be called with template syntax directly. Use wrapper functions:

```cpp
// CCE built-in signature
get_buf(pipe, buf_id, mode);   // Acquire buffer
rls_buf(pipe, buf_id, mode);   // Release buffer

// Wrapper template for cleaner syntax
template <int pipe>
AICORE void get_buffer(int id)
{
    get_buf(pipe, id, 0);
}

template <int pipe>
AICORE void rls_buffer(int id)
{
    rls_buf(pipe, id, 0);
}

// Usage
get_buffer<PIPE_MTE2>(buf_id);
TLOAD(...);
rls_buffer<PIPE_MTE2>(buf_id);
```

**Pros:** No priming/draining needed, single ID for RAW+WAR deps
**Cons:** Requires careful placement to ensure correct ordering

## Test Results

| Sync Method | Cycles | Max Diff | Status |
|-------------|--------|----------|--------|
| set_flag/wait_flag | 24,276 | 9.54e-06 | ✅ PASSED |
| get_buf/rls_buf | 23,803 | 12.6 | ❌ FAILED |

**Note:** Buffer-ID sync currently produces incorrect results. Investigation needed on:
- Correct mode parameter values
- Proper placement relative to instructions
- Whether additional sync is needed between stages

## Build & Run

```bash
# Build
source /usr/local/Ascend/cann_9b2/cann/set_env.sh
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
