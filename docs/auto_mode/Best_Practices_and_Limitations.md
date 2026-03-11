# Auto Mode Best Practices and Limitations

## Scope

This document outlines the current limitations in auto mode. It addresses what coding practices should and should not be followed in order for auto synchronization and memory allocation to work as expected.

## Auto Synchronization

Here are some patterns that should be avoided (and their alternative) to make sure auto synchronization works as intended in auto mode.

### if-statements in Loops That Will Only Run on Either the First or Last Iterations Should be Peeled Out of the Loop

Here is an example of this that shouldn't be done in auto mode:

```cpp
...

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    if (tile_id == 0) {
        TLOAD(srcTile, globalSrc);
    }

    ...

    if (tile_id == total_tiles-1) {
        TSTORE(globalDst, dstTile);
    }
}

...
```

Those branches should instead be pulled out of the loop:

```cpp
...

TLOAD(srcTile, globalSrc);

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {

    ...

}

TSTORE(globalDst, dstTile);

...
```

### if-statements Nested in a Loop That Don't Depend on Loop Variables Should be Pulled Out of the Loop

For example, consider the if-statement inside the inner loop:

```cpp
...

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...

    for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
        if (next_tile != -1) {
            ... // computation here
        }
    }
}

...
```

This should instead be rewritten as

```cpp
...

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    int next_tile = tile_id < total_tiles-1 ? tile_id + 1 : -1;
    ...

    if (next_tile != -1) {
        for (int subtile_id = 0; subtile_id < total_subtiles; subtile_id++) {
            ... // computation here
        }
    }
}

...
```

### PTO Instructions After a Loop Should be Moved After an if-statement Instead

Consider this example:

```cpp
...

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    ...
}

TSTORE(globalDst, dstTile);

if (DAV_VEC) {
    TLOAD(srcTile, globalSrc);
}

...
```

The `TSTORE` right after the loop should instead be moved after the if-statement.

```cpp
...

for (int tile_id = 0; tile_id < total_tiles; tile_id++) {
    ...
}

if (DAV_VEC) {
    TLOAD(srcTile, globalSrc);
}

TSTORE(globalDst, dstTile); // now after if-statement

...
```

### Evaluate Complex if-statement Conditions Before Using it in an if-statement

Consider the following if-statement with multiple ands/ors in the condition:

```cpp

if ((srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned()) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}

```

The condition should be evaluated before instead.

```cpp
bool cond = (srcTile.GetValidRow() > 16 || srcTile.GetValidCol() > 16) && srcTile.GetKAligned();

if (cond) {
    TLOAD(srcTile, globalSrc1);
}
else {
    TLOAD(srcTile, globalSrc0);
}

```

## Memory Allocation


In Auto mode, the compiler will assign a constant memory address to each declared tile variable. That is, unlike in manual mode, we cannot have tiles that change their memory address in the middle of the kernel. For example, consider the following PTO Manual code:

```cpp
...

TileData tile;

for (int i = 0; i < N; i++) {
    TASSIGN(tile, 0x100 * i);
    foo(tile);
}

...
```

This cannot be replicated in auto mode since the address of `tile` changes throughout the kernel.

To account for this limitation, we have provided two PTO Auto instructions that let users specify some level of memory address dependency between tiles:

* `TRESHAPE`

    The Auto Mode has a different implementation of `TRESHAPE` to the Manual Mode. In Manual Mode, it assigns the address of the "source" tile to the "destination" tile. On the other hand, in Auto Mode, `TRESHAPE` acts as a compiler hint that tells the compiler to assign the destination tile the same address as the source tile.

* `TSUBVIEW`

    `TSUBVIEW` lets users get a subtile from a larger tile. The compiler will then calculate the relative offset of the subtile and add it to the allocated address of the source tile.

Since the address of tiles are constant throughout the kernel, we cannot change the address of a tile with `TRESHAPE` or `TSUBVIEW`. For example,
```cpp
...

TRESHAPE(tile0, tile1);
foo(tile0);
TSUBVIEW(tile0, tile2, 0, 0);
bar(tile0);

...
```

is not allowed. Multiple `TRESHAPE` or `TSUBVIEW` calls with the same destination tile will lead to a compiler crash.

These operations only make sense if they are placed right after the declaration of the destination tile. Otherwise, we will still treat uses of the tile before the `TRESHAPE` or `TSUBVIEW` calls as though they already take the address dependency of the source tile. For example,

```cpp
...

foo(tile0); // tile0 already takes the address of tile1 here.
TRESHAPE(tile0, tile1);

...
```