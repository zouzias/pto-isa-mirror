# PTO AS Documentation Guide

This page is the main entry for PTO AS documentation. It helps readers quickly locate assembly-related documents by topic instead of navigating individual files one by one.

PTO AS documentation mainly covers the following areas:

- PTO-AS syntax, grammar, and textual representation
- ISA-level tile operations and auxiliary AS constructs
- Scalar arithmetic and control-flow operations reused from MLIR
- Assembly-related conventions and supporting references

## Recommended Reading Path

If you are new to PTO-AS, we recommend reading in the following order:

1. [PTO-AS Specification](PTO-AS.md): understand the textual format, syntax, and directives
2. [PTO AS Operations Reference](README.md): get an overview of operation categories and linked references
3. [PTO-AS Conventions](conventions.md): understand naming and documentation conventions
4. Operation category documents: read the category pages relevant to your task

## Documentation Categories

### 1. PTO-AS Syntax and Core Specification

- [PTO-AS Specification](PTO-AS.md): textual format, SSA-style naming, directives, and grammar overview
- [PTO-AS Conventions](conventions.md): assembly syntax conventions and related documentation rules
- `PTO-AS.bnf`: formal BNF grammar definition for PTO-AS

### 2. PTO Tile Operation Categories

- [Elementwise Operations](elementwise-ops.md): tile-tile elementwise operations
- [Tile-Scalar Operations](tile-scalar-ops.md): tile-scalar arithmetic, comparison, and activation operations
- [Axis Reduction and Expansion](axis-ops.md): row/column reductions and broadcast-like expansion operations
- [Memory Operations](memory-ops.md): GM and tile data movement operations
- [Matrix Multiplication](matrix-ops.md): GEMM and GEMV related operations
- [Data Movement and Layout](data-movement-ops.md): extraction, insertion, transpose, reshape, and padding operations
- [Complex Operations](complex-ops.md): sorting, gather/scatter, random, quantization, and utility operations
- [Manual Resource Binding](manual-binding-ops.md): assignment and hardware/resource configuration operations

### 3. Auxiliary AS and MLIR-Derived Operations

- [Auxiliary Functions](nonisa-ops.md): tensor views, tile allocation, indexing, and synchronization helpers
- [Scalar Arithmetic Operations](scalar-arith-ops.md): scalar-only arithmetic operations from MLIR `arith`
- [Control Flow Operations](control-flow-ops.md): structured control-flow operations from MLIR `scf`

### 4. Related References

- [ISA Instruction Reference](../isa/README.md): canonical per-instruction semantics
- [docs Entry Guide](../README.md): top-level documentation navigation for PTO Tile Lib

## Directory Structure

Key entries are listed below:

```text
├── PTO-AS*                     # PTO-AS syntax and specification documents
├── conventions*                # Assembly conventions documents
├── elementwise-ops*            # Elementwise tile operation references
├── tile-scalar-ops*            # Tile-scalar operation references
├── axis-ops*                   # Axis reduction and expansion references
├── memory-ops*                 # Memory operation references
├── matrix-ops*                 # Matrix multiplication references
├── data-movement-ops*          # Data movement and layout references
├── complex-ops*                # Complex operation references
├── manual-binding-ops*         # Manual resource binding references
├── scalar-arith-ops*           # Scalar arithmetic references
├── control-flow-ops*           # Control-flow references
└── nonisa-ops*                 # Auxiliary AS construct references
```

## Related Entry Points

- [ISA Instruction Reference](../isa/README.md): browse canonical PTO instruction semantics
- [docs Entry Guide](../README.md): return to the main docs navigation page
- [Machine Documentation](../machine/README.md): understand the abstract execution model
