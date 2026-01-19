from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class DType(str, Enum):
    f32 = "f32"
    f16 = "f16"
    i32 = "i32"
    i16 = "i16"
    i8 = "i8"
    u8 = "u8"


@dataclass(frozen=True)
class TileType:
    rows: int
    cols: int
    dtype: DType

    def mlir(self) -> str:
        return f"!pto.tile<{self.rows}x{self.cols}x{self.dtype.value}>"

    def as_tilebuf(self) -> "TileBufType":
        return TileBufType(rows=self.rows, cols=self.cols, dtype=self.dtype)


@dataclass(frozen=True)
class TileBufType:
    rows: int
    cols: int
    dtype: DType

    def mlir(self) -> str:
        return f"!pto.tilebuf<{self.rows}x{self.cols}x{self.dtype.value}>"


@dataclass(frozen=True)
class MemRefType:
    space: str  # e.g. "gm"
    shape: Tuple[int, int]
    dtype: DType

    def tile(self) -> TileType:
        r, c = self.shape
        return TileType(r, c, self.dtype)

    def tilebuf(self) -> TileBufType:
        r, c = self.shape
        return TileBufType(r, c, self.dtype)

    def mlir(self) -> str:
        r, c = self.shape
        return f"!pto.memref<{self.space},{r}x{c}x{self.dtype.value}>"
