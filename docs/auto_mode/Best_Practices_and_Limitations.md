# Auto Mode Programming Best Practices and Restrictions

## Scope
To achieve best performance, kernel developers who use the PTO AUTO Mode need to follow a number of practices and restrictions. In most cases, not following these rules will still results in correct compilation, but performance of the generated kernel may be impacted.  

## Control Flow Rules 
Complex control flow (especially inside loops) often makes it difficult to optimize the precise cross-pipe parallelization and double-buffering. Since PTO AUTO compiler is required to maintain program correctness, it may generate the synchronization operations more conservatively resulting in performance degradation.  

### Guards for First and Last Iteraions 
Any condition that guards the first and last iteraion of a loop should be expressed in a form that can be statically evaluated. That makes it possible for the PTO AUTO compiler to automatically peel the first and last iteration of the loop which results in simplifying the auto synchronization substantially.  Here is an example: 

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

### Loop Invariant Control Flow in Loop Nests  
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


### Complex Logical Expressions in If Statements  

It is highly recommended that complex logical expressions that gaurd PTO instructions are evaluated prior to being used in the if-statement. As an example, consider the following if-statement:
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

## Memory Allocation Rules 

PTO AUTO compiler automatically assigns constant memory addresses to each declared tile variable. That means, unlike in PTO manual mode, tile memory addresses cannot change in the middle of a kernel. For example, consider the following PTO Manual code:

```cpp
TileData tile;

for (int i = 0; i < N; i++) {
    TASSIGN(tile, 0x100 * i);
    foo(tile);
}
```

This logic cannot be expressed in PTO AUTO since the address of `tile` changes throughout the kernel.

To address this limitation, PTO AUTO uses two instructions, ``TRESHAPE`` and ``TSUBVIEW``, that allow users to specify memory address dependency between tiles.

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

