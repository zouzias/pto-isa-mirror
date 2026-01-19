//===- PTOTypeDefs.td - PTO type definitions ----------------*- tablegen -*-===//
//
// PTO-AS uses MLIR-like type spellings such as `!pto.tile<...>` and
// `!pto.memref<...>`. The PTO-AS spec treats these spellings as opaque.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOTYPEDEFS
#define MLIR_DIALECT_PTO_IR_PTOTYPEDEFS

include "mlir/Dialect/PTO/IR/PTODialect.td"
include "mlir/IR/AttrTypeBase.td"

// Opaque tile type: `!pto.tile<...>`
def PTO_TileType : TypeDef<PTO_Dialect, "Tile"> {
  let mnemonic = "tile";
  let summary = "Opaque on-chip tile type";
  let description = [{
    PTO-AS treats tile type payloads as opaque (e.g. `64x64xf32`, `...`).
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque tile buffer type: `!pto.tilebuf<...>`
//
// This models a *mutable* tile register / buffer used by destination-passing
// style (DPS) ops. The payload is intentionally opaque and may match the
// `!pto.tile<...>` spelling for convenience.
def PTO_TileBufType : TypeDef<PTO_Dialect, "TileBuf"> {
  let mnemonic = "tilebuf";
  let summary = "Opaque mutable tile buffer type";
  let description = [{
    PTO-AS treats tile buffer payloads as opaque (e.g. `64x64xf32`, `...`).
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque tile view type: `!pto.tileview<...>`
//
// A view descriptor for a tile-shaped region (view-only, no allocation).
def PTO_TileViewType : TypeDef<PTO_Dialect, "TileView"> {
  let mnemonic = "tileview";
  let summary = "Opaque tile view type";
  let description = [{
    PTO-AS treats tile views as opaque. The payload (if any) is target-defined.
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque memref/global view type: `!pto.memref<...>`
def PTO_MemRefType : TypeDef<PTO_Dialect, "MemRef"> {
  let mnemonic = "memref";
  let summary = "Opaque global memory view type";
  let description = [{
    PTO-AS treats memref type payloads as opaque (e.g. `gm,64x64xf32`, `gm,...,f32`).
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque tensor/global view type: `!pto.tensor<...>`
//
// This is a semantic alias for global memory views used by some frontends.
// PTO-AS currently standardizes on `!pto.memref<...>`; both spellings are
// treated as opaque payloads by the PTO toolchain.
def PTO_TensorType : TypeDef<PTO_Dialect, "Tensor"> {
  let mnemonic = "tensor";
  let summary = "Opaque global tensor view type";
  let description = [{
    PTO-AS treats tensor type payloads as opaque (e.g. `gm,64x64xf32`, `gm,...,f32`).
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque tensor view descriptor: `!pto.tensor_view<...>`
//
// This is intended for frontends that want to model pointers + shape/stride
// without committing to `memref` semantics.
def PTO_TensorViewType : TypeDef<PTO_Dialect, "TensorView"> {
  let mnemonic = "tensor_view";
  let summary = "Opaque tensor view descriptor type";
  let description = [{
    PTO-AS treats tensor views as opaque. The payload (if any) is target-defined.
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

// Opaque event type: `!pto.event<...>`
def PTO_EventType : TypeDef<PTO_Dialect, "Event"> {
  let mnemonic = "event";
  let summary = "Opaque event type";
  let description = [{
    PTO-AS treats events as opaque. The payload (if any) is target-defined.
  }];
  let parameters = (ins StringRefParameter<"opaque payload">:$spec);
  let hasCustomAssemblyFormat = 1;
}

#endif // MLIR_DIALECT_PTO_IR_PTOTYPEDEFS
