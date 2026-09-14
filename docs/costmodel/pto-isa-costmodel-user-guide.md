# PTO-ISA Costmodel User Guide

This document describes how to enable the PTO-ISA host costmodel for A2/A3 and A5 targets.

The costmodel build is a host-side build path. User code still includes PTO-ISA headers and calls PTO APIs such as `TADD`, `TLOAD`, `TSTORE`, and `SYNCALL`, but the implementation is redirected to host mocks. The costmodel records PTO instructions and returns estimated cycle counts without launching real NPU kernels.

## Supported targets

| Target | `ARCH` value | `__NPU_ARCH__` | Cycle source |
| --- | --- | --- | --- |
| A2/A3 | `A2A3` | `2201` | lightweight costmodel and perf_sim fallback |
| A5 | `A5` | `3101` | Existing memory, synchronization, and cube models; VF formulas are being added incrementally |

## CMake integration

Include the PTO costmodel helper and enable it on the target that builds the host test or tool:

```cmake
include(<pto-isa-root>/include/pto/costmodel/cmake/pto_costmodel.cmake)

add_executable(my_costmodel_test main.cpp)
pto_enable_costmodel(my_costmodel_test ARCH A2A3)
```

For A5:

```cmake
include(<pto-isa-root>/include/pto/costmodel/cmake/pto_costmodel.cmake)

add_executable(my_a5_costmodel_test main.cpp)
pto_enable_costmodel(my_a5_costmodel_test ARCH A5)
```

`pto_enable_costmodel()` adds the PTO include directory and defines:

```text
__COSTMODEL
PTO_COMM_NOT_SUPPORTED
__NPU_ARCH__=<target arch>
```

A5 uses only the headers shipped with PTO-ISA. It does not require simulator sources, an LLVM pass, or an extra library.

## A2/A3 behavior

In A2/A3 mode, PTO wrappers are compiled through the costmodel path:

```text
pto/pto-inst.hpp
  -> costmodel/runtime_stub.hpp
  -> costmodel/pto_instr.hpp
  -> a2a3/cce_costmodel/*
```

The A2/A3 costmodel uses the lightweight formula model where available. If an instruction is not covered by the lightweight model, perf_sim uses a fallback estimate based on opcode, tile shape, data type, and pipe stage.

## A5 behavior

In A5 mode, PTO wrappers are compiled through the A5 costmodel path:

```text
pto/pto-inst.hpp
  -> costmodel/runtime_stub.hpp
  -> costmodel/pto_instr.hpp
  -> a5/cce_costmodel/*
```

A lightweight host mock supplies declarations and no-op implementations for device intrinsics used by A5 implementation headers. VF cycles will be predicted directly by per-tileop formulas; the host path no longer captures a VF instruction stream or builds an intermediate representation.

Phase 1 does not yet provide concrete VF formulas. A VF instruction is therefore recorded as `Unsupported`, its cycle value remains zero, and a diagnostic is attached. The generic fallback is not used to fabricate a valid VF estimate. Memory, synchronization, and cube instructions continue to use their existing models.

## Reading cycle results

The costmodel records PTO instructions through perf_sim. Typical users should consume the recorded cycle values from the costmodel/perf_sim result path used by their test or integration.

For A5, check `costmodel_status` before consuming a cycle value. A cycle is a valid prediction only when the status is `Supported`; an `Unsupported` record contains a diagnostic naming the VF instruction that is not covered yet.

## Notes

- The costmodel path is for host-side estimation and validation. It does not execute real NPU kernels.
- The A5 host mock only lets a standard C++ compiler parse device intrinsics; it does not execute device operations.
- Use the same PTO APIs as the normal NPU path; select the costmodel path through CMake definitions instead of changing user kernel code.
