# Auto Mode

## Scope

This section gives an overview of PTO auto mode.

## What is Auto Mode

Auto mode does all the memory allocation for tiles and synchronization in the compiler. Programming in auto mode works just like in manual mode, except there is no need for `TASSIGN` and `TSYNC` (in fact, these will do nothing in auto mode). Auto mode is targetted towards users that don't want to worry about allocating tile buffers and managing pipelines 

## Auto Mode Limitations/Best Practices

### Auto Synchronization

Here are some patterns that should be avoided to make sure auto synchronization works as intended in auto mode.

## Examples