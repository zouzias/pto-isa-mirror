# PTO Costmodel Proposal

## Overview

This document proposes `include/pto/costmodel` as a host-side PTO costmodel backend for PTO-ISA review.

Its purpose is to:

- compile PTO NPU code on a standard host toolchain
- record PTO and CCE execution traces
- estimate cost from those traces
- keep costmodel-specific logic isolated from the main PTO common path

This backend is intended for analysis and evaluation. It is not a numerical simulator.

## Purpose

`include/pto/costmodel` is designed to provide a lightweight analysis path for PTO developers.

It addresses four practical needs:

- Host compilation: allow PTO code that normally depends on device qualifiers, ACL runtime APIs, and CCE intrinsics to compile on host.
- Trace visibility: expose which top-level PTO APIs execute and which low-level CCE calls they emit.
- Cost estimation: convert trace output into architecture-aware cycle estimates.
- Clean integration: keep the main PTO include flow unchanged while concentrating costmodel logic in a dedicated folder.

## Realization Method

The design works by inserting a host-side compatibility and tracing backend behind the standard PTO include entry point.

### Overall Flow

```text
                    user code includes pto/pto-inst.hpp
                                  |
                     +------------+------------+
                     |                         |
                     | __COSTMODEL disabled    | __COSTMODEL enabled
                     |                         |
                     v                         v
               normal PTO path        pto/costmodel/runtime_stub.hpp
                                                |
                           +--------------------+--------------------+
                           |                    |                    |
                           v                    v                    v
                common/qualifiers.hpp   common/aclrt_stub.hpp  common/runtime_util.hpp
                - blank device attrs    - fake ACL runtime     - fake flags/helpers
                - host-safe macros      - malloc/memcpy/free   - trap/min/max/etc.
                                                |
                                                v
                                  common/arch_select.hpp
                                                |
                            +-------------------+-------------------+
                            |                                       |
                            v                                       v
                    a2a3/cce_stub.hpp                        a5/cce_stub.hpp
                    - fake CCE stubs                         - fake CCE stubs
                    - RecordCceCall(...)                    - RecordCceCall(...)
                            |                                       |
                            +-------------------+-------------------+
                                                |
                                                v
                                    costmodel/pto_instr.hpp
                              - wraps top-level PTO APIs in scope
                              - records PTO instruction names
                                                |
                                                v
                                         trace.hpp
                               - PTO records
                               - CCE call records
                               - arguments
                                                |
                                                v
                            evaluator/trace_evaluator.hpp + arch_config.hpp
                               - rule lookup by arch
                               - cycle estimation
```

### Integration Point

The user-facing include remains `pto/pto-inst.hpp`.

When `__COSTMODEL` is enabled, [`pto-inst.hpp`](/home/lc/pto-isa-costmodel/include/pto/pto-inst.hpp) routes the build to:

- [`runtime_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/runtime_stub.hpp)
- [`pto_instr.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/pto_instr.hpp)

This keeps the external programming model unchanged and makes costmodel a backend selection rather than a separate API.

### Host Runtime Compatibility

The host runtime layer is implemented by:

- [`common/qualifiers.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/qualifiers.hpp)
- [`common/aclrt_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/aclrt_stub.hpp)
- [`common/runtime_util.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/runtime_util.hpp)

These files replace device-side assumptions with host-safe definitions.

```text
device-side expectation                costmodel replacement
-----------------------                ---------------------
__gm__ / __ubuf__ / AICORE            empty host-safe macros
ACL runtime APIs                      malloc/free/memcpy stubs
flag/wait helpers                     host helper shims
CCE tile pointer access               host-visible pointer path
```

The key idea is to preserve the original PTO logic while faking the surrounding runtime environment needed for host analysis.

### Trace Capture

Trace capture is centralized in [`trace.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/trace.hpp).

It records:

- top-level PTO instruction names
- emitted CCE calls
- key call arguments
- raw CCE calls outside PTO grouping when needed

At the PTO API boundary, [`costmodel/pto_instr.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/pto_instr.hpp) wraps top-level PTO APIs with `PtoInstrScope`.

This creates a grouped execution record:

```text
PTO call on host
   |
   v
PtoInstrScope("TLOAD")
   |
   v
original PTO implementation runs
   |
   v
fake CCE stubs record calls such as:
- copy_gm_to_ubuf_align_*
- vadd / vmul / vsub
- mad
- copy_ubuf_to_gm_align_*
   |
   v
trace grouped under TLOAD / TADD / TMATMUL / TSTORE
```

### Architecture-Specific Realization

Architecture dispatch is handled by [`common/arch_select.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/common/arch_select.hpp):

```text
__NPU_ARCH__ == 2201        -> a2a3/cce_stub.hpp
__NPU_ARCH__ == 3101/3510   -> a5/cce_stub.hpp
```

Current structure:

- [`a2a3/cce_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a2a3/cce_stub.hpp) provides broad fake intrinsic coverage for A2/A3 PTO flows.
- [`a5/cce_stub.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/cce_stub.hpp) plus [`a5/compat.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/compat.hpp) provide a controlled A5-compatible subset.

This separates shared host infrastructure from arch-specific stub behavior and makes backend expansion incremental.

### Latency Evaluation

The evaluator layer is implemented by:

- [`arch_config.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/arch_config.hpp)
- [`evaluator/cce_evaluator.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/evaluator/cce_evaluator.hpp)
- [`evaluator/trace_evaluator.hpp`](/home/lc/pto-isa-costmodel/include/pto/costmodel/evaluator/trace_evaluator.hpp)

This layer applies explicit architecture rules to the trace and produces cycle estimates.

```text
recorded CCE call
   |
   v
lookup rule in ArchConfig
   |
   +--> fixed rule      -> constant cycles
   +--> repeat rule     -> startup + repeats * unit cost
   +--> burst rule      -> startup + burst/length cost
   +--> mad rule        -> tile-based matrix estimate
   |
   v
CCE call cycle estimate
   |
   v
aggregate into PTO instruction total
   |
   v
aggregate into whole trace total
```

## Merits

The primary merit is separation of concerns. Costmodel logic stays in [`include/pto/costmodel`](/home/lc/pto-isa-costmodel/include/pto/costmodel) instead of spreading across the main PTO common implementation. This improves maintainability and reduces interference with normal PTO development.

The second merit is development efficiency. Developers can compile and inspect PTO lowering behavior on host without requiring a real device runtime. This is useful for instruction bring-up, trace inspection, and costmodel tuning.

The third merit is architectural clarity. The implementation is cleanly divided into:

- host runtime compatibility
- arch-specific intrinsic stubs
- trace recording
- latency evaluation

The fourth merit is compatibility with the normal PTO programming model. User code still enters through `pto/pto-inst.hpp`, so costmodel behaves as a backend path rather than a separate framework.

## Usage

The intended usage flow is:

```text
1. include pto/pto-inst.hpp
2. build with __COSTMODEL enabled
3. select arch through __NPU_ARCH__
4. run PTO code on host
5. collect PTO + CCE trace
6. evaluate trace into cycle estimates
```

Current documentation and demos are located at:

- [`README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/README.md)
- [`a2a3/README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a2a3/README.md)
- [`a5/README.md`](/home/lc/pto-isa-costmodel/include/pto/costmodel/a5/README.md)
- [`tests/costmodel`](/home/lc/pto-isa-costmodel/tests/costmodel)

In practice, this backend is used to answer questions such as:

- which PTO path is actually executed
- which CCE operations are emitted
- which arguments affect those operations
- what approximate cycle cost follows from the trace

## Summary

`include/pto/costmodel` should be understood as a host-side PTO trace and cost-evaluation backend. It preserves the normal PTO front door, realizes costmodel behavior through host runtime substitution plus traceable intrinsic stubs, and provides a structured way to convert PTO execution traces into architecture-aware cycle estimates.
