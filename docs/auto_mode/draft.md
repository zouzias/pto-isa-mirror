# Auto Mode

## Scope

This section gives an overview of PTO auto mode.

## What is Auto Mode

Auto mode does all the memory allocation for tiles and synchronization in the compiler. Programming in auto mode works just like in manual mode, except there is no need for `TASSIGN` and `TSYNC` (in fact, these will do nothing in auto mode). Auto mode is targetted towards users that don't want to worry about allocating tile buffers and managing pipelines, focusing on the actual computation instead.

## Compiling Auto Mode with Ascend CANN

TODO

## Auto Mode Limitations/Best Practices

### Auto Synchronization

Here are some patterns that should be avoided (and their alternative) to make sure auto synchronization works as intended in auto mode.

* **if-statements in loops that will only run on either the first or last iterations should be peeled out of the loop.**

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

* **if-statements nested in a loop that don't depend on the loop variable should be pulled out of the loop.**

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

* **PTO instructions after a loop should be moved after an if-statement instead.**

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

## Examples