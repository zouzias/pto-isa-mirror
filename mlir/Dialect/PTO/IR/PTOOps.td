//===- PTOOps.td - Pattern descriptor operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTO dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPS
#define MLIR_DIALECT_PTO_IR_PTOOPS

include "mlir/Dialect/PTO/IR/PTODialect.td"
include "mlir/Dialect/PTO/IR/PTOAttrs.td"
include "mlir/Dialect/PTO/IR/PTOTypeDefs.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def PTOTileLike : Type<CPred<
    "(::llvm::isa<::mlir::pto::TileType>($_self) || ::llvm::isa<::mlir::pto::TileBufType>($_self) || ::llvm::isa<::mlir::pto::TileViewType>($_self))">,
    "PTO tile-like type (`!pto.tile`, `!pto.tilebuf`, `!pto.tileview`)">;

// Backward-compatible alias used by legacy op definitions in this file.
def PTOTile : PTOTileLike;

def PTOMemRef : Type<CPred<"::llvm::isa<::mlir::pto::MemRefType>($_self)">,
                     "PTO memref/global view type">;

def PTOTensor : Type<CPred<"::llvm::isa<::mlir::pto::TensorType>($_self)">,
                     "PTO tensor/global view type">;

def PTOGlobalView : Type<CPred<
    "(::llvm::isa<::mlir::pto::MemRefType>($_self) || ::llvm::isa<::mlir::pto::TensorType>($_self))">,
    "PTO global view type (`!pto.memref` or `!pto.tensor`)">;

def PTOEvent : Type<CPred<"::llvm::isa<::mlir::pto::EventType>($_self)">,
                    "PTO event type">;

def PTOTensorView : Type<CPred<"::llvm::isa<::mlir::pto::TensorViewType>($_self)">,
                         "PTO tensor view descriptor type">;

def TensorOrMemref : AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

def PTODpsType : AnyTypeOf<[AnyRankedTensor, AnyMemRef, PTOGlobalView, PTOTensorView, PTOTileLike],
                           "PTO DPS operand type">;

//===----------------------------------------------------------------------===//
// Base op class
//===----------------------------------------------------------------------===//

class PTO_Op<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

class PTO_DpsOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [DestinationStyleOpInterface])> {
  code extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Generic tile-op templates (ISA-style)
//===----------------------------------------------------------------------===//

class PTO_TileUnaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_DpsOp<mnemonic, traits> {
  let arguments = (ins PTOTileLike:$src, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

class PTO_TileBinaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_DpsOp<mnemonic, traits> {
  let arguments = (ins PTOTileLike:$lhs, PTOTileLike:$rhs, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

class PTO_TileTernaryOp<string mnemonic, list<Trait> traits = []>
    : PTO_DpsOp<mnemonic, traits> {
  let arguments = (ins PTOTileLike:$src0, PTOTileLike:$src1, PTOTileLike:$src2, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:` type($src0) `,` type($src1) `,` type($src2) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

class PTO_TileScalarOp<string mnemonic, list<Trait> traits = []>
    : PTO_DpsOp<mnemonic, traits> {
  let arguments = (ins PTOTileLike:$src, AnyType:$scalar, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` type($src) `,` type($scalar) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Memory (GM <-> Tile)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Pointer/View Ops (front-end IR)
//===----------------------------------------------------------------------===//

def MakeTensorViewOp : PTO_Op<"make_tensor_view", [AttrSizedOperandSegments]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (no allocation, no copy)";

  let arguments = (ins
    AnyType:$ptr,
    Variadic<Index>:$shape,
    Variadic<Index>:$strides
  );

  let results = (outs PTOTensorView:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SubviewOp : PTO_Op<"subview"> {
  let summary = "Subview on tensor_view to tileview (view-only, no copy)";
  let arguments = (ins
    PTOTensorView:$source,
    Variadic<Index>:$offsets,
    I64ArrayAttr:$sizes
  );

  let results = (outs PTOTileLike:$result);

  let assemblyFormat = [{
    $source `,` `offsets` `=` `[` $offsets `]` `,` `sizes` `=` $sizes
    attr-dict `:` type($source) `->` type($result)
  }];
}

def AllocTileOp : PTO_Op<"alloc_tile"> {
  let summary = "ALLOC_TILE - Allocate a tile buffer (logical register/buffer)";
  let arguments = (ins);
  let results = (outs PTOTileLike:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def TLoadOp : PTO_DpsOp<"tload"> {
  let summary = "TLOAD - Load data from a global memref view into a tile";
  let arguments = (ins PTODpsType:$src_mem, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src_mem `:` type($src_mem) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def TStoreOp : PTO_DpsOp<"tstore"> {
  let summary = "TSTORE - Store a tile into a global memref view";
  let arguments = (ins PTOTileLike:$src, PTODpsType:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def TStoreFpOp : PTO_DpsOp<"tstore.fp"> {
  let summary = "TSTORE_FP - Store an accumulator tile using a scaling tile";
  let arguments = (ins PTOTileLike:$src, PTOTileLike:$fp, PTOGlobalView:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` type($src) `,` type($fp) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def MgatherOp : PTO_DpsOp<"mgather"> {
  let summary = "MGATHER - Gather-load from global memory using per-element indices";
  let arguments = (ins PTOGlobalView:$mem, PTOTileLike:$idx, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $mem `,` $idx `:` type($mem) `,` type($idx) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def MscatterOp : PTO_DpsOp<"mscatter"> {
  let summary = "MSCATTER - Scatter-store to global memory using per-element indices";
  let arguments = (ins PTOTileLike:$src, PTOTileLike:$idx, PTOGlobalView:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` type($src) `,` type($idx) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Element access
//===----------------------------------------------------------------------===//

def GetValOp : PTO_Op<"getval"> {
  let summary = "GETVAL - Read a single tile element into a scalar";
  let arguments = (ins PTOTileLike:$src, Index:$offset);
  let results = (outs AnyType:$dst);
  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` type($src) `,` type($offset) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def SetValOp : PTO_Op<"setval"> {
  let summary = "SETVAL - Write a scalar into a single tile element";
  let arguments = (ins PTOTileLike:$dst, Index:$offset, AnyType:$val);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $dst `,` $offset `,` $val `:` type($dst) `,` type($offset) `,` type($val) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise (Tile-Tile)
//===----------------------------------------------------------------------===//

def TAbsOp  : PTO_TileUnaryOp<"tabs">;
def TNegOp  : PTO_TileUnaryOp<"tneg">;
def TNotOp  : PTO_TileUnaryOp<"tnot">;
def TExpOp  : PTO_TileUnaryOp<"texp">;
def TLogOp  : PTO_TileUnaryOp<"tlog">;
def TSqrtOp : PTO_TileUnaryOp<"tsqrt">;
def TRSqrtOp: PTO_TileUnaryOp<"trsqrt">;
def TRecipOp: PTO_TileUnaryOp<"trecip">;
def TReluOp : PTO_TileUnaryOp<"trelu">;
def TTransOp: PTO_TileUnaryOp<"ttrans">;
def TFillPadOp : PTO_TileUnaryOp<"tfillpad">;

def TAddOp  : PTO_TileBinaryOp<"tadd">;
def TSubOp  : PTO_TileBinaryOp<"tsub">;
def TMulOp  : PTO_TileBinaryOp<"tmul">;
def TDivOp  : PTO_TileBinaryOp<"tdiv">;
def TRemOp  : PTO_TileBinaryOp<"trem">;
def TShlOp  : PTO_TileBinaryOp<"tshl">;
def TShrOp  : PTO_TileBinaryOp<"tshr">;
def TAndOp  : PTO_TileBinaryOp<"tand">;
def TOrOp   : PTO_TileBinaryOp<"tor">;
def TXorOp  : PTO_TileBinaryOp<"txor">;
def TMinOp  : PTO_TileBinaryOp<"tmin">;
def TMaxOp  : PTO_TileBinaryOp<"tmax">;

def TAddcOp : PTO_TileTernaryOp<"taddc">;
def TSubcOp : PTO_TileTernaryOp<"tsubc">;

// Parametric ReLU with per-element slope: `tprelu %src, %slopeTile`.
def TPReluOp : PTO_TileBinaryOp<"tprelu">;

// Per-element select: `tsel %mask, %src0, %src1`.
def TSelOp : PTO_DpsOp<"tsel"> {
  let summary = "TSEL - Per-element selection using a mask tile";
  let arguments = (ins PTOTileLike:$mask, PTOTileLike:$src0, PTOTileLike:$src1, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $mask `,` $src0 `,` $src1 `:` type($mask) `,` type($src0) `,` type($src1) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

// Compare ops (packed predicate mask)
def TCmpOp : PTO_DpsOp<"tcmp"> {
  let summary = "TCMP - Compare two tiles and produce a packed predicate tile";
  let arguments = (ins PTOTileLike:$src0, PTOTileLike:$src1, PTOTileLike:$dst, OptionalAttr<PTO_CmpModeAttr>:$cmpMode);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` type($src0) `,` type($src1) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def TCmpsOp : PTO_DpsOp<"tcmps"> {
  let summary = "TCMPS - Compare a tile against a scalar and produce a packed predicate tile";
  let arguments = (ins PTOTileLike:$src, AnyType:$scalar, PTOTileLike:$dst, OptionalAttr<PTO_CmpModeAttr>:$cmpMode);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` type($src) `,` type($scalar) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

// Type conversion with rounding mode.
def TCvtOp : PTO_DpsOp<"tcvt"> {
  let summary = "TCVT - Elementwise type conversion with rounding mode";
  let arguments = (ins PTOTileLike:$src, PTOTileLike:$dst, OptionalAttr<PTO_RoundModeAttr>:$rmode);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Tile-Scalar / Tile-Immediate
//===----------------------------------------------------------------------===//

def TAddsOp  : PTO_TileScalarOp<"tadds">;
def TSubsOp  : PTO_TileScalarOp<"tsubs">;
def TMulsOp  : PTO_TileScalarOp<"tmuls">;
def TDivsOp  : PTO_TileScalarOp<"tdivs">;
def TRemsOp  : PTO_TileScalarOp<"trems">;
def TMaxsOp  : PTO_TileScalarOp<"tmaxs">;
def TMinsOp  : PTO_TileScalarOp<"tmins">;
def TAndsOp  : PTO_TileScalarOp<"tands">;
def TOrsOp   : PTO_TileScalarOp<"tors">;
def TXorsOp  : PTO_TileScalarOp<"txors">;
def TLReluOp : PTO_TileScalarOp<"tlrelu">;

def TExpandsOp : PTO_DpsOp<"texpands"> {
  let summary = "TEXPANDS - Broadcast a scalar into a tile";
  let arguments = (ins AnyType:$scalar, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $scalar `:` type($scalar) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

// Global select: `tsels %src0, %src1, %selectMode`.
def TSelsOp : PTO_DpsOp<"tsels"> {
  let summary = "TSELS - Global select between two tiles using a scalar selectMode";
  let arguments = (ins PTOTileLike:$src0, PTOTileLike:$src1, AnyType:$selectMode, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $selectMode `:` type($src0) `,` type($src1) `,` type($selectMode) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def TAddscOp : PTO_DpsOp<"taddsc"> {
  let summary = "TADDSC - Fused add: src0 + scalar + src1";
  let arguments = (ins PTOTileLike:$src0, AnyType:$scalar, PTOTileLike:$src1, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:` type($src0) `,` type($scalar) `,` type($src1) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

def TSubscOp : PTO_DpsOp<"tsubsc"> {
  let summary = "TSUBSC - Fused op: src0 - scalar + src1";
  let arguments = (ins PTOTileLike:$src0, AnyType:$scalar, PTOTileLike:$src1, PTOTileLike:$dst);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:` type($src0) `,` type($scalar) `,` type($src1) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Axis Reduce / Expand
//===----------------------------------------------------------------------===//

def TRowSumOp : PTO_TileUnaryOp<"trowsum">;
def TRowMaxOp : PTO_TileUnaryOp<"trowmax">;
def TRowMinOp : PTO_TileUnaryOp<"trowmin">;
def TRowExpandOp : PTO_TileUnaryOp<"trowexpand">;
def TColMaxOp : PTO_TileUnaryOp<"tcolmax">;
def TColMinOp : PTO_TileUnaryOp<"tcolmin">;
def TColExpandOp : PTO_TileUnaryOp<"tcolexpand">;

def TRowExpandDivOp : PTO_TileBinaryOp<"trowexpanddiv">;
def TRowExpandMulOp : PTO_TileBinaryOp<"trowexpandmul">;
def TRowExpandSubOp : PTO_TileBinaryOp<"trowexpandsub">;

def TColSumOp : PTO_Op<"tcolsum"> {
  let summary = "TCOLSUM - Reduce each column by summing across rows";
  let arguments = (ins PTOTile:$dst, PTOTile:$src, DefaultValuedOptionalAttr<BoolAttr, "false">:$isBinary);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src attr-dict `:` type($dst) `,` type($src) `->` `()`
  }];
}

//===----------------------------------------------------------------------===//
// Matrix Multiply
//===----------------------------------------------------------------------===//

def TMatmulOp : PTO_Op<"tmatmul"> {
  let summary = "TMATMUL - Matrix multiply producing an output tile";
  let arguments = (ins PTOTile:$dst, PTOTile:$a, PTOTile:$b);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $a `,` $b attr-dict `:` `(` type($dst) `,` type($a) `,` type($b) `)` `->` `()`
  }];
}

def TMatmulAccOp : PTO_Op<"tmatmul.acc"> {
  let summary = "TMATMUL_ACC - Matrix multiply with accumulator input";
  let arguments = (ins PTOTile:$dst, PTOTile:$acc, PTOTile:$a, PTOTile:$b);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $acc `,` $a `,` $b attr-dict
      `:` `(` type($dst) `,` type($acc) `,` type($a) `,` type($b) `)` `->` `()`
  }];
}

def TMatmulBiasOp : PTO_Op<"tmatmul.bias"> {
  let summary = "TMATMUL_BIAS - Matrix multiply with bias add";
  let arguments = (ins PTOTile:$dst, PTOTile:$a, PTOTile:$b, PTOTile:$bias);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $a `,` $b `,` $bias attr-dict
      `:` `(` type($dst) `,` type($a) `,` type($b) `,` type($bias) `)` `->` `()`
  }];
}

def TMatmulMxOp : PTO_Op<"tmatmul.mx"> {
  let summary = "TMATMUL_MX - Matrix multiply with scaling tiles";
  let arguments = (ins PTOTile:$dst, PTOTile:$a, PTOTile:$a_scale, PTOTile:$b, PTOTile:$b_scale);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $a `,` $a_scale `,` $b `,` $b_scale attr-dict
      `:` `(` type($dst) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)` `->` `()`
  }];
}

def TMatmulMxAccOp : PTO_Op<"tmatmul.mx.acc"> {
  let summary = "TMATMUL_MX (acc) - Matrix multiply with scaling tiles and accumulator input";
  let arguments = (ins PTOTile:$c_out, PTOTile:$c_in, PTOTile:$a, PTOTile:$a_scale, PTOTile:$b, PTOTile:$b_scale);
  let results = (outs);
  let assemblyFormat = [{
    $c_out `,` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale attr-dict
      `:` `(` type($c_out) `,` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)` `->` `()`
  }];
}

def TMatmulMxBiasOp : PTO_Op<"tmatmul.mx.bias"> {
  let summary = "TMATMUL_MX (bias) - Matrix multiply with scaling tiles and bias";
  let arguments = (ins PTOTile:$dst, PTOTile:$a, PTOTile:$a_scale, PTOTile:$b, PTOTile:$b_scale, PTOTile:$bias);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias attr-dict
      `:` `(` type($dst) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` type($bias) `)` `->` `()`
  }];
}

//===----------------------------------------------------------------------===//
// Data Movement / Layout / Complex
//===----------------------------------------------------------------------===//

def TAssignOp : PTO_Op<"tassign"> {
  let summary = "TASSIGN - Bind a tile to an implementation-defined address";
  let arguments = (ins PTOTile:$tile, Index:$addr);
  let results = (outs);
  let assemblyFormat = [{
    $tile `,` $addr attr-dict `:` type($tile) `,` type($addr)
  }];
}

// `tmov.<mode>` family (see docs/isa/TMOV.md)
def TMovM2LOp : PTO_TileUnaryOp<"tmov.m2l">;
def TMovM2ROp : PTO_TileUnaryOp<"tmov.m2r">;
def TMovM2BOp : PTO_TileUnaryOp<"tmov.m2b">;
def TMovM2SOp : PTO_TileUnaryOp<"tmov.m2s">;
def TMovA2VOp : PTO_TileUnaryOp<"tmov.a2v">;
def TMovV2VOp : PTO_TileUnaryOp<"tmov.v2v">;

def TMovFpOp : PTO_Op<"tmov.fp"> {
  let summary = "TMOV_FP - Move with scaling tile";
  let arguments = (ins PTOTile:$dst, PTOTile:$src, PTOTile:$fp);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src `,` $fp attr-dict `:` `(` type($dst) `,` type($src) `,` type($fp) `)` `->` `()`
  }];
}

def TReshapeOp : PTO_TileUnaryOp<"treshape">;

def TExtractOp : PTO_Op<"textract"> {
  let summary = "TEXTRACT - Extract a sub-tile from a source tile";
  let arguments = (ins PTOTile:$dst, PTOTile:$src, Index:$r0, Index:$r1);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src `[` $r0 `,` $r1 `]` attr-dict
      `:` `(` type($dst) `,` type($src) `,` type($r0) `,` type($r1) `)` `->` `()`
  }];
}

// TCI
def TCIOp : PTO_Op<"tci"> {
  let summary = "TCI - Generate a contiguous integer sequence into a tile";
  let arguments = (ins PTOTile:$dst, AnyType:$S, DefaultValuedOptionalAttr<BoolAttr, "false">:$descending);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $S attr-dict `:` `(` type($dst) `,` type($S) `)` `->` `()`
  }];
}

// TGATHER (index or mask-pattern)
def TGatherOp : PTO_Op<"tgather"> {
  let summary = "TGATHER - Gather/select using indices tile or maskPattern attribute";
  let arguments = (ins PTOTile:$dst, PTOTile:$src0, Optional<PTOTile>:$indices,
                   OptionalAttr<PTO_MaskPatternAttr>:$maskPattern);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src0 (`,` $indices^)? attr-dict `:` type($dst) `,` type($src0) `->` `()`
  }];
}

def TGatherBOp : PTO_Op<"tgatherb"> {
  let summary = "TGATHERB - Gather elements using byte offsets";
  let arguments = (ins PTOTile:$dst, PTOTile:$src, PTOTile:$offsets);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src `,` $offsets attr-dict `:` type($dst) `,` type($src) `,` type($offsets) `->` `()`
  }];
}

def TScatterOp : PTO_Op<"tscatter"> {
  let summary = "TSCATTER - Scatter rows using per-element row indices";
  let arguments = (ins PTOTile:$dst, PTOTile:$src, PTOTile:$idx);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $src `,` $idx attr-dict `:` `(` type($dst) `,` type($src) `,` type($idx) `)` `->` `()`
  }];
}

def TSort32Op : PTO_Op<"tsort32"> {
  let summary = "TSORT32 - Sort a block and produce an index mapping";
  let arguments = (ins PTOTile:$dst, PTOTile:$idx, PTOTile:$src);
  let results = (outs);
  let assemblyFormat = [{
    $dst `,` $idx `,` $src attr-dict `:` `(` type($dst) `,` type($idx) `,` type($src) `)` `->` `()`
  }];
}

def TMrgsortOp : PTO_Op<"tmrgsort"> {
  let summary = "TMRGSORT - Merge sort for multiple sorted lists";
  let arguments = (ins PTOTile:$dst, PTOTile:$src0, PTOTile:$src1,
                   DefaultValuedOptionalAttr<BoolAttr, "false">:$exhausted);
  let results = (outs AnyType:$executed);
  let assemblyFormat = [{
    $dst `,` $src0 `,` $src1 attr-dict `:`
      `(` type($dst) `,` type($src0) `,` type($src1) `)` `->` type($executed)
  }];
}
def TPartAddOp : PTO_TileBinaryOp<"tpartadd">;
def TPartMaxOp : PTO_TileBinaryOp<"tpartmax">;
def TPartMinOp : PTO_TileBinaryOp<"tpartmin">;

// Debug printing (tile or global view).
def TPrintOp : PTO_Op<"tprint"> {
  let summary = "TPRINT - Debug print a tile or global view";
  let arguments = (ins AnyType:$src);
  let results = (outs);
  let assemblyFormat = [{
    $src attr-dict `:` type($src)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TSyncOp : PTO_Op<"tsync"> {
  let summary = "TSYNC - Synchronize PTO execution using event operands";
  let arguments = (ins PTOEvent:$e0, PTOEvent:$e1);
  let results = (outs);
  let assemblyFormat = [{
    $e0 `,` $e1 attr-dict `:` type($e0) `,` type($e1)
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOOPS
