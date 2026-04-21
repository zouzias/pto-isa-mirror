# cube_matmul_nbuf — N-Buffer Cube MatMul Benchmark

## Overview

Benchmarks the ping-pong / N-buffer pattern for cube matrix multiply on A5
(Ascend 950B /  simulator).

**All configs:** 

| Config | K_tile | #Bufs | L1 slots | L0 slots | Buf IDs used |
|--------|--------|-------|----------|----------|--------------|
|   | 16 | 2 | 0–1   | 2–3   | 5  |
|   | 16 | 4 | 0–3   | 4–7   | 9  |
|   | 16 | 8 | 0–7   | 8–15  | 17 |
|  | 32 | 2 | 0–1   | 2–3   | 5  |
|  | 32 | 4 | 0–3   | 4–7   | 9  |

---

## GM Memory Layout — Normal Row-Major

A single  /  /  set is shared by **all 5 configs**
(both K_tile=16 and K_tile=32 variants).



### How the kernel accesses tiles



This is the standard GEMM convention.  K_TILE=16 and K_TILE=32 share the
same binary files because the stride in the GlobalTensor description handles
the non-contiguous column slice of A.

---

## GlobalTensor Shape/Stride

# 0 "<stdin>"
# 0 "<built-in>"
# 0 "<command-line>"
# 1 "/usr/include/stdc-predef.h" 1 3 4
# 0 "<command-line>" 2
# 1 "<stdin>"

---

## TASSIGN (L1/L0 buffer slot addresses)

| Memory | K_tile=16 stride | K_tile=32 stride | Notes |
|--------|-----------------|-----------------|-------|
| L1 A   | 0x800 (2 KB)    | 0x1000 (4 KB)   | NZ padded |
| L1 B   | 0x2000 (8 KB)   | 0x4000 (16 KB)  | |
| L0A    | 0x400 (1 KB)    | 0x800 (2 KB)    | |
| L0B    | 0x2000 (8 KB)   | 0x4000 (16 KB)  | |

---

## Building and Running

target_dir: /home/happy/.openclaw/workspace/tests/npu/a5/src/st
target_dir: /home/happy/.openclaw/workspace/tests/npu/a5/src/st
target_dir: /home/happy/.openclaw/workspace/tests/npu/a5/src/st
clean build: build
build failed: -- The C compiler identification is GNU 15.2.0
-- The CXX compiler identification is GNU 15.2.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /home/linuxbrew/.linuxbrew/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/linuxbrew/.linuxbrew/bin/g++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
CMake Error at CMakeLists.txt:22 (message):
  Cannot find ASCEND_HOME_PATH, please run set_env.sh.


-- Configuring incomplete, errors occurred!

Expected: all 5 configs PASS (max diff < 1e-2, bad count = 0).

---

## Files

| File | Description |
|------|-------------|
|  | Kernel — 5 templated configs |
|  | GTest harness — 5 test cases |
|  | Data generator — normal row-major layout |
|  | Build target  |
|  | This file |
