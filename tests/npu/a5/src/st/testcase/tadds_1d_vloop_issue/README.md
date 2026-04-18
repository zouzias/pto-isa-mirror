# TADDS 1D RV_VLOOP Hardware Loop Issue

## Problem

On A5, TADDS 1D generates RV_VLDS (scalar register offset) instead of RV_VLDI (immediate offset),
preventing the compiler from using RV_VLOOP hardware loops.

## Benchmark (1x4096 float32, 10 iterations)

| Op    | Ticks  | RV_VLOOP | RVECSU | EPC  |
|-------|--------|----------|--------|------|
| TADD  | 21,799 | 10       | 50     | 1.88 |
| TADDS | 34,253 | 0        | 9,160  | 1.20 |

TADDS is 1.57x slower due to 183x more scalar instructions.

## Instruction Evidence

TADD: `RV_VLDI Vd[0], Sn[65]=0x100, #offset=8`
TADDS: `RV_VLDS Vd[0], Sn[8]=0x0, Sn2[9]=0x0, Sm[70]=0x100`

## Environment

- CANN 9.0.0-beta.2, A5 simulator (dav_3510)
- Compiler: ccec clang 15.0.5
