//===- PTODialect.td - PTO dialect --------------------------*- tablegen -*-===//
//
// This file declares the PTO dialect used to model PTO-AS / PTO ISA operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTODIALECT
#define MLIR_DIALECT_PTO_IR_PTODIALECT

include "mlir/IR/DialectBase.td"

def PTO_Dialect : Dialect {
  let name = "pto";
  let cppNamespace = "::mlir::pto";
  let summary = "PTO (Programmable Tensor Operations) dialect";
  let description = [{
    PTO models the PTO-AS instruction set over tile values (e.g. `!pto.tile<...>`)
    and opaque global memory views (e.g. `!pto.memref<...>`), with ISA-like
    mnemonics such as `pto.tload`, `pto.tadd`, and `pto.tmatmul`.
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTODIALECT

