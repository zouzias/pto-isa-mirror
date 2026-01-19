# 6. PTO Assembly language (PTO-AS)

## 6.1 Scope

PTO-AS is the assembly syntax for the PTO ISA.

This manual defines the *shape* of PTO-AS programs and how they map to the ISA operand model. The normative grammar and conventions are in:

- PTO-AS spec: `docs/grammar/PTO-AS.md`
- Conventions: `docs/grammar/conventions.md`

## 6.2 Core ideas

PTO-AS syntax is intentionally designed for:

- explicit operand typing
- explicit tile locations and modifiers
- easy parsing/printing (assembler/disassembler tooling)

## 6.3 Operand classes (conceptual)

PTO-AS typically uses operand classes such as:

- tile operands (with location/type/shape metadata)
- global operands (GlobalTensor views)
- scalar immediates / registers (modifiers and modes)

Exact syntax depends on the instruction; see per-instruction pages (for example `docs/isa/TLOAD.md` and `docs/isa/TSTORE.md`).

## 6.4 Examples

Load–compute–store (schematic, instruction-centric):

```asm
// t0, t1, tout are Vec tiles
TLOAD   t0, g0
TLOAD   t1, g1
TADD    tout, t0, t1
TSTORE  gout, tout
```

Synchronization (schematic):

```asm
TSYNC   TLOAD, TADD
TSYNC   TADD,  TSTORE
```

MLIR-compatible `.pto` example (dialect-qualified ops):

```text
module {
  func.func @main(%a: !pto.memref<gm,64x64xf32>, %b: !pto.memref<gm,64x64xf32>, %out: !pto.memref<gm,64x64xf32>) {
    %c0 = arith.constant 0 : index
    %t0 = pto.tload %a[%c0, %c0] : (!pto.memref<gm,64x64xf32>, index, index) -> !pto.tile<64x64xf32>
    %t1 = pto.tload %b[%c0, %c0] : (!pto.memref<gm,64x64xf32>, index, index) -> !pto.tile<64x64xf32>
    %t2 = pto.tadd %t0, %t1 : (!pto.tile<64x64xf32>, !pto.tile<64x64xf32>) -> !pto.tile<64x64xf32>
    pto.tstore %out[%c0, %c0], %t2 : (!pto.memref<gm,64x64xf32>, index, index, !pto.tile<64x64xf32>) -> ()
    return
  }
}
```

For real examples and constraints, see:

- `docs/grammar/PTO-AS.md`
- `docs/isa/README.md`
