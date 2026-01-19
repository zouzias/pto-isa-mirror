//===- PTOAttrs.td - PTO attribute definitions --------------*- tablegen -*-===//
//
// PTO-AS uses dialect attributes for ISA modifiers such as:
//   - #pto.cmp<EQ>
//   - #pto.round_mode<CAST_RINT>
//   - #pto.mask_pattern<P0101>
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOATTRS
#define MLIR_DIALECT_PTO_IR_PTOATTRS

include "mlir/Dialect/PTO/IR/PTODialect.td"
include "mlir/IR/EnumAttr.td"

// Compare modes used by `tcmp`/`tcmps`.
def PTO_CmpModeAttr : I32EnumAttr<"CmpMode", "Comparison mode", [
  I32EnumAttrCase<"EQ", 0>,
  I32EnumAttrCase<"NE", 1>,
  I32EnumAttrCase<"LT", 2>,
  I32EnumAttrCase<"LE", 3>,
  I32EnumAttrCase<"GT", 4>,
  I32EnumAttrCase<"GE", 5>
]> {
  let cppNamespace = "::mlir::pto";
  let mnemonic = "cmp";
}

// Rounding modes used by `tcvt`.
def PTO_RoundModeAttr : I32EnumAttr<"RoundMode", "Rounding mode", [
  I32EnumAttrCase<"CAST_NONE", 0>,
  I32EnumAttrCase<"CAST_RINT", 1>,
  I32EnumAttrCase<"CAST_ROUND", 2>,
  I32EnumAttrCase<"CAST_FLOOR", 3>,
  I32EnumAttrCase<"CAST_CEIL", 4>,
  I32EnumAttrCase<"CAST_TRUNC", 5>,
  I32EnumAttrCase<"CAST_ODD", 6>
]> {
  let cppNamespace = "::mlir::pto";
  let mnemonic = "round_mode";
}

// Mask patterns used by `tgather` (mask-pattern form).
def PTO_MaskPatternAttr : I32EnumAttr<"MaskPattern", "Mask pattern", [
  I32EnumAttrCase<"P0101", 1>,
  I32EnumAttrCase<"P1010", 2>,
  I32EnumAttrCase<"P0001", 3>,
  I32EnumAttrCase<"P0010", 4>,
  I32EnumAttrCase<"P0100", 5>,
  I32EnumAttrCase<"P1000", 6>,
  I32EnumAttrCase<"P1111", 7>
]> {
  let cppNamespace = "::mlir::pto";
  let mnemonic = "mask_pattern";
}

#endif // MLIR_DIALECT_PTO_IR_PTOATTRS

