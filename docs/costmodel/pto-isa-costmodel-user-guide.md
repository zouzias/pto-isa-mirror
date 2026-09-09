# PTO-ISA Costmodel User Guide

This document describes how to enable the PTO-ISA host costmodel for A2/A3 and A5 targets.

The costmodel build is a host-side build path. User code still includes PTO-ISA headers and calls PTO APIs such as `TADD`, `TLOAD`, `TSTORE`, and `SYNCALL`, but the implementation is redirected to host mocks. The costmodel records PTO instructions and returns estimated cycle counts without launching real NPU kernels.

## Supported targets

| Target | `ARCH` value | `__NPU_ARCH__` | Cycle source |
| --- | --- | --- | --- |
| A2/A3 | `A2A3` | `2201` | lightweight costmodel and perf_sim fallback |
| A5 | `A5` | `3101` | A5 VfSim for VF scopes, plus perf_sim/fallback for other instructions |

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

For A5 it also enables the VfSim build support and links the required VfSim components.

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
  -> pkg_inc/pto/costmodel/vfsim/*
```

A5 vector instructions inside a VF scope are captured as `VfInfo` and passed to VfSim. VfSim returns cycle counts, and the PTO costmodel records those cycles as part of the current PTO instruction.

Memory, synchronization, cube, and unsupported VF forms are handled by host mocks and fallback estimates. Fallback is intended to keep cycle prediction available when a detailed VfSim result cannot be produced.

## Reading cycle results

The costmodel records PTO instructions through perf_sim. Typical users should consume the recorded cycle values from the costmodel/perf_sim result path used by their test or integration.

For A5, the public cycle path is intentionally cycle-oriented. Fallback may be used internally, but packaging users are expected to rely on the returned cycle count rather than detailed fallback logs.

## Notes

- The costmodel path is for host-side estimation and validation. It does not execute real NPU kernels.
- A5 costmodel builds require the VfSim CMake helper to be available under `pkg_inc/pto/costmodel/vfsim/cmake`.
- Use the same PTO APIs as the normal NPU path; select the costmodel path through CMake definitions instead of changing user kernel code.
