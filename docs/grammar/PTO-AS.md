# PTO-AS (PTO Assembly) Specification

PTO-AS is a textual, instruction-centric assembly format for PTO Tile Lib. It is designed to be:

- close to the PTO instruction set (`TADD`, `TLOAD`, `TMATMUL`, ...),
- readable and easy to diff (one instruction per line),
- compatible with MLIR tooling (SSA value naming, MLIR-like type spellings, and a direct mapping to an MLIR textual form).

PTO-AS is designed to be consumed/produced by an MLIR-based assembler/disassembler.

In this repository, we additionally define a **PTO MLIR textual form** that is **syntactically MLIR** (i.e. it is a
`module { func.func ... }` with `pto.*` operations). The `.pto` files produced by the Python demo frontend use this
form.

## 1. High-Level Form

### 1.1 Instruction form

A PTO-AS program is a list of statements. The most common statement is an instruction:

```text
%dst = tadd %src0, %src1 : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> !pto.tile<32x32xf32>;
```

The toolchain also supports a **register-style** destination-passing form (DPS), where the destination tile is an
explicit operand and the instruction has no SSA result:

```text
tadd %dst, %src0, %src1 : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> ();
```

PTO-AS uses SSA-like value names (`%dst`, `%src0`) to stay close to MLIR’s assembly conventions; this keeps the
format deterministic and makes it easy to round-trip through MLIR bytecode.

PTO-AS is a synchronous, line-ordered format: there is no `wait(...)` clause and no implicit event result. If a program
needs to model an explicit dependency, it uses an explicit instruction (for example `tsync`) with event operands.

Operands may also include indexed forms (commonly used by memory ops):

```text
%t0 = tload %sv[%c0, %c1] : (!pto.memref<...>, index, index) -> !pto.tile<...>;
```

Type signatures (`: ...`) are recommended for readability but may be omitted when the types are unambiguous in context.

### 1.2 MLIR-compatible `.pto` files (recommended)

To make PTO programs parseable by MLIR tooling, write PTO-AS *inside an MLIR module* and use dialect-qualified operation
names (`pto.*`):

```text
module {
  func.func @main(%a: !pto.memref<gm,64x64xf32>, %b: !pto.memref<gm,64x64xf32>, %out: !pto.memref<gm,64x64xf32>) {
    %c0 = arith.constant 0 : index
    %t0 = pto.alloc_tile : !pto.tilebuf<64x64xf32>
    pto.tload ins(%a[%c0, %c0] : !pto.memref<gm,64x64xf32>) outs(%t0 : !pto.tilebuf<64x64xf32>)
    %t1 = pto.alloc_tile : !pto.tilebuf<64x64xf32>
    pto.tload ins(%b[%c0, %c0] : !pto.memref<gm,64x64xf32>) outs(%t1 : !pto.tilebuf<64x64xf32>)
    %t2 = pto.alloc_tile : !pto.tilebuf<64x64xf32>
    pto.tadd ins(%t0, %t1 : !pto.tilebuf<64x64xf32>, !pto.tilebuf<64x64xf32>) outs(%t2 : !pto.tilebuf<64x64xf32>)
    pto.tstore ins(%t2 : !pto.tilebuf<64x64xf32>) outs(%out[%c0, %c0] : !pto.memref<gm,64x64xf32>)
    return
  }
}
```

Notes:

- `pto.*` operation names are MLIR dialect-qualified.
- Types are still printed using MLIR-style spellings (for example `!pto.tile<...>`).
- The demo tooling accepts both SSA-result and DPS/register forms and internally normalizes them.
- For correctness across pipelines (MTE/Vector/etc), the demo assembler (`ptoas`) inserts conservative synchronization
  between major pipeline transitions.

## 2. Types

PTO-AS uses MLIR-like type spellings:

- Tile values: `!pto.tile<...>` (opaque)
- Global memory / views: `!pto.memref<...>` (opaque)
- Events: `!pto.event` (opaque)
- Scalars: MLIR builtin types like `index`, `i32`, `f32`

The assembler treats these as *opaque* types; they are carried through bytecode but not semantically verified unless a
target-specific verifier is introduced later.

## 3. Attributes

Instruction modifiers that are not positional operands (e.g., compare modes) are written as an MLIR-style attribute
dictionary:

```text
%mask = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

## 4. Directives

PTO-AS supports a small set of non-instruction directives for declaring external inputs and constants.

Argument declaration (introduces an SSA value):

```text
.arg %a : !pto.tile<16x16xf16>;
```

Event arguments (when modeling a dependency explicitly):

```text
.arg %e0 : !pto.event;
```

Constant declaration (introduces an SSA value):

```text
.const %c0 = 0 : index;
```

## 5. Grammar

The normative grammar is provided in:

- `docs/grammar/PTO-AS.bnf`
