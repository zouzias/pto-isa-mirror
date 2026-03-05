
This file lists some rules that must be followed by kernel developers (i.e., programmers that use auto-mode compiler and pto-isa library to write their own kernels).

Not following these rules can lead to any of the following consequences:

1. Fail to compile (either from source code level or crash in the compiler)
2. Functionally incorrect (e.g., precision issues)
3. Bad performance

# 1 - Control Flow Rules

Complex control flow (especially inside loops) often makes it difficult to optimize the precise cross-pipe parallelization and double-buffering. Since PTO AUTO compiler is required to maintain program correctness, it may generate the sync operations more conservatively leading to performance degradation.  

## 1.1 - Guards for First and Last Iteraions

Any condition that guards the first and last iteraion of a loop should be expressed in a form that can be statically evaluated. That makes it possible for the PTO AUTO compiler to automatically peel peel the first and last iteration of the loop which results in simplifying the auto synchronization substantially.  Here is an example:

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    if (tile_id == 0) {
        TLOAD(srcTile, globalSrc);
    }
    ...
    if (tile_id == total_tiles-1) {
        TSTORE(globalDst, dstTile);
    }
}
```

## 1.2 - Loop Invariant Control Flow in Loop Nests  

if-statements Nested in a Loop that do not depend on inner loop's induction variable should be left inside the inner loop.  

For example, consider the if-statement inside the inner loop:

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...
    for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
        if (next_tile != -1) {
            ... // computation here
        }
    }
}
```

This should instead be rewritten as

```cpp
for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...
    if (next_tile != -1) {
        for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
            ... // computation here
        }
    }
}
```

## 1.3 - Complex Logical Expressions in If Statements

It is highly recommended that complex logical expressions that guard PTO instructions are evaluated prior to being used in the if-statement.

As an example, consider the following if-statement:

```cpp
if ((srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned()) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}
```

It is recommended to be rewritten to the following form:

```cpp
bool cond = (srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned();

if (cond) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}
```

# 2 - Memory Allocation Rules

## 2.1 Use `TRESHAPE` to tell compiler that 2 tiles have the same base address

If you want to express that tile B has the same base address as tile A (alias), you can use `TRESHAPE` which works for both manual and auto mode, instead of using `TASSIGN` directly. For instance:

```cpp
TileSrcTypeA tileA;
TileSrcTypeB tileB;

// Invalid in auto mode
TASSIGN(tileA, 0x0);
TASSIGN(tileB, 0x0);

// Correct in auto mode
TRESHAPE(tileB, tileA);
```

## 2.2 Use `TSUBVIEW` to tell compiler that tile B is a sub-view of tile A ***

Same purpose as `TRESHAPE`, but this is used to express that tile B's address is based on tile A's address, plus some offset. You can check `docs/isa/TSUBVIEW.md` for details.

Example:

```cpp
uint16_t rowOffset, colOffset; // can be runtime variable

// addr(tileB) = addr(tileA) + offsets
TileData tileA(...);
TileData tileB(...);

// Invalid in auto mode
TASSIGN(tileA, 0x0);
TASSIGN(tileB, 0x0 + rowOffset * TileData::Col + colOffset * 1 + sizeof(T));

// Correct for auto mode
TSUBVIEW(tileB, tileA, rowOffset, colOffset);
```

## 2.3 - Keep in mind that tile's memory shouldn't change at runtime in auto mode

PTO AUTO compiler automatically assigns constant memory addresses to each declared tile variable. That means, unlike in PTO manual mode, tile memory addresses cannot change in the middle of a kernel. For example, consider the following PTO Manual code:

```cpp
TileData tile;
for (int i = 0; i < N; i++) {
    TASSIGN(tile, 0x100 * i);
    foo(tile);
}
```

Users in manual mode can use trick like this to change a tile' address at any point during runtime. Compiler can't handle such dynamic behavior to properly allocate memory; *In auto mode, each tile will be allocated memory once and for all.* Programmers should rewrite their code to bypass such limitation.

Here's a crucial mindset for auto mode:  
**think of a tile as a C++ reference, meaning that its memory address is already determined and cannot change once declared**.

## 2.4 - Correctly understand the semantics of `TRESHAPE` and `TSUBVIEW`

In manual mode, they are both actual PTO instructions that (re)assign an address to a tile. However, their semantic is different in auto mode: **they simply serve as a hint to the compiler about how 2 tiles alias with each other.**

* `TRESHAPE`:

    The semantic of the `TRESHAPE` instruction is slightly different in the AUTO mode compared to the Manual Mode. In Manual Mode, ``TRESHAPE`` it assigns the address of the ``source`` tile to the ``destination`` tile at the point of execution of the ``TRESHAPE`` instruction. However, in the AUTO mode, `TRESHAPE` acts as a mechanism to bind the source and destination tile to the same address. This binding is valid across the entire scope in which the source and destination tiles are defined.

* `TSUBVIEW`:

    The `TSUBVIEW` instruction allows users obtain a subtile from a larger tile. In the AUTO mode, the compiler calculates the relative offset of the subtile and add it to the automatically allocated address of the base tile.

Note that since in the AUTO mode is that the address of a tile cannot change throughout its scope, a tile cannot be used as the destination for multiple `TRESHAPE` or `TSUBVIEW` instructions. For example, the following example is invalid in the AUTO mode and the actual behavior is undefined.  

```cpp
TRESHAPE(tile0, tile1);
foo(tile0);
... 
TSUBVIEW(tile0, tile2, 0, 0);
bar(tile0);
```

As a good practice, it is highly recommended that the ``TSUBVIEW`` and ``TRESHAPE`` instructions are placed right the declaration of the destination tiles.

# 3 - Rules to handle aliasing scenarios (temporary)

In last section, we have introduced `TRESHAPE` and `TSUBVIEW` to express aliasing of 2 tiles; however, the actual rules to use them are somewhat complex at this point. As a current limitation of compiler, **it is strongly recommended NOT to alias tiles at this point.**

**NOTE: this section only provides temporary hacks to solve the aliasing issue, due to an inherent flaw in the compiler IR design; once the new, redesigned IR of the compiler is merged, this section isn't needed.**

## 3.1 - Rules for using `TRESHAPE` and `TSUBVIEW`

Signature:

```cpp
TRESHAPE(aliasTile, originalTile)
TSUBVIEW(aliasTile, originalTile, rowOffset, colOffset)
```

Use `TRESHAPE` to assign the `originalTile`'s address to `aliasTile`.
You can think of an equivalent C++ semantics like this:

```cpp
Tile *aliasTile = &originalTile;
```

For `TSUBVIEW`, you can think of it as:

```cpp
Tile *aliasTile = calculate_address(&originalTile, rowOffset, colOffset);
```

However, you must follow these 2 rules to use both:
(1) They must be placed before the first use of `aliasTile`.
(2) The first use of `aliasTile` should be used only as input tile; if it's used as an output, then you can't use them, you should use `TALIAS` instead (see `TALIAS` section below for details).

Use `TRESHAPE` as example (same applies to `TSUBVIEW`):

```cpp
// we want tileSrc1 to be assgned with the same address as tileSrc0
TileSrcData tileSrc0, tileSrc1;
TileDstData tileDst;

// CORRECT: placed before first use of tileSrc1, and first use of tileSrc1 is used as input
TRESHAPE(tileSrc1, tileSrc0);
TADD(tileDst, tileSrc0, tileSrc1);

// WRONG: placed after first use of tileSrc1
TADD(tileDst, tileSrc0, tileSrc1);
TRESHAPE(tileSrc1, tileSrc0);

// WRONG: placed before first use of tileSrc1, but first use of tileSrc1 is used as output.
// Should use TALIAS instead
TRESHAPE(tileSrc1, tileSrc0);
TEXP(tileSrc1, tileSrc0);
```

There's one exception for `TSUBVIEW`: the `aliasTile` can't be used as output such as `TALIAS` does (see `TALIAS` section below).

## 3.2 - Rules for using `TALIS`

Signature: `TALIAS(originalTile, aliasTile)`

Serve exactly the same purpose as `TRESHAPE`, but used in different cases. Note that the parameters are reversed compared to `TRESHAPE`.
You must follow these 2 rules to use `TALIAS`:

1. It must be placed after the first use of `aliasTile`.
2. The first use of `aliasTile` is used as an output tile; if it's used as an input, you should use `TRESHAPE` instead.

Example:

```cpp
// we want tileDst to be assigned with the same address as tileSrc
TileSrcData tileSrc;
TileDstData tileDst;

// CORRECT: placed after first use of tileDst, and first use of tileDst is used as output
TEXP(tileDst, tileSrc);
TALIAS(tileSrc, tileDst);

// WRONG: placed before the first use of tileDst
TALIAS(tileSrc, tileDst);
TEXP(tileDst, tileSrc);

// WRONG: placed after first use of tileDst, but first use of tileDst is used as input.
// Should use TRESHAPE instead
TEXP(tileSrc, tileDst);
TALIAS(tileSrc, tileDst);
```

## 3.3 - Rule when the same tile is used as outputs for multiple PTO instructions

In the source code, it's common that a tile is used as output for multiple PTO instructions. Currently, a temporary hack is needed to bypass the issue for compiler: **you need to create a new tile for each of these outputs, and use `TALIAS` to alias them back to the original tile.**

Example:

```cpp
// WRONG: the same tile used as output, causing issue to compiler
TADD(tileDst, ...);
TEXP(tileDst, ...);
TROWSUM(tileDst, ...);

// CORRECT: create new tile for each of these outputs,
// and call TALIAS to map them back
TADD(tileDst, ...);

TileDataDst tileDstTmp1;
TEXP(tileDstTmp1, ...);
TALIAS(tileDst, tileDstTmp1);

TileDataDst tileDstTmp2;
TROWSUM(tileDstTmp2, ...);
TALIAS(tileDst, tileDstTmp2);
```

# 4 - General rules

## 4.1 - Don't call `TLOAD` on a destination tile

If a tile is supposed to be used as output and has no copied-in data from GM, you should never call `TLOAD` on it.
First: it's unnecessary;
Second: it may cause data races in auto mode.

Take this example:

```cpp
TLOAD(dstTile, dstGlobal); // redundant
TLOAD(srcTile, srcGlobal);

TEXP(dstTile, srcTile);

TSTORE(dstGlobal, dstTile);
```

In manual mode, programmers can assign `srcTile` and `dstTile` with different addresses manually, and this works perfectly fine.

However, in auto mode, this may be problematic: here the liveness range of `dstTile` and `srcTile` won't overlap,
thus the compiler may reuse the memory by assigning them to the same address. If they're both called with `TLOAD`,
then these 2 `TLOAD` will occur at the same time and as a consequence, they overwrite each other.

DON'T CALL REDUNDANT `TLOAD`!

## 4.2 - Don't call CCE intrinsics directly

Kernel developers using PTO should call PTO instructions only; they shouldn't
call CCE intrinsics directly. There are 2 reasons:

1. CCE intrinsics take raw pointer as arguments, which won't compile for auto Mode (since tile is represented as a vector type instead of pointer type in auto mode; see `pto-isa/include/pto/common/memory.hpp` for details).
2. Automatic memory allocation and synchronization is built upon the analysis of PTO instructions only; they can't recognize anything else.

For this reason, you shouldn't call `.data()` member function of a tile directly in your kernel; technically speaking, this is NOT an interface to be called by the kernel developers, it should only be used for library developers instead.

If you have no choice (e.g., there's no equivalent of a PTO instruction) but use CCE intrinsics directly, you should summit a request to pto-isa to add new PTO instruction.

## 4.3 - Prefer using `PtoSetWaitFlag` instead of `set_flag` and `wait_flag`

Internal implementation of `PtoSetWaitFlag` has guards against manual and auto mode; when compiling in auto mode, this interface is a no-op, so that it doesn't conflict with auto-sync from compiler. This can save you some typing and make the code look cleaner.

If you call `set_flag` and `wait_flag` directly in your kernel, you always need to manually guard it using the macro `__PTO_AUTO__`, which is cumbersome.
