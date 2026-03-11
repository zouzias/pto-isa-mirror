# PTO AUTO Mode 

## What is PTO AUTO

PTO AUTO is a programming mode for PTO that that greatly simplifies developing efficient PTO code while providing kernel developers with the mechanisms that are necessary to implement their optimizations. More specifically, in PTO AUTO, the kernel developer does not need to explicitly specify tile memory addresses or synchronization between different pipes. Instead the PTO AUTO compiler automatically allocates optimal memory addressess for the tiles in different chip buffers. Moreover, the compiler automatically synchronizes the PTO tile operations in order to maximize parallelism among different pipes. 



## Auto Mode Features
. Programming in auto mode works just like in manual mode, except there is no need for `TASSIGN` and `TSYNC`/`Event` (in fact, these will do nothing in auto mode). 

The following sections will explain the features that auto mode offers such that it enables users to write in such a programming model as an alternative.

### Automated Tile Live Range Analysis

In auto mode, PTO compilation will keep track of each Tile and its live-ranges. This is a core component that is used as analysis to serve the following features mentioned below.

### Automatic Synchronization
 	 
In manual mode, user would normally have to keep track of the asynchronous nature of the hardware by using PTO's [`event model`](../coding/Event.md) at precise code locations in order to ensure both functional correctness and high performance in execution. This might be tedious and error prone.

Auto mode compilation will allow users to avoid having to use the event model to synchronize their code. The compiler will automatically determine the locations to insert synchronization under the hood - ensuring functional correctness and competitive performance.

### Tile Memory Allocation
 	 
In the default mode of PTO compilation, after instantiating `Tile` variables, we would need to complement them with a `TASSIGN` instruction to manually assign a dedicated buffer address that it operates on. However in auto mode, this is not required anymore. By simply instantiating the `Tile` variable the compiler will automatically allocate the buffer addresses under the hood for the user.

## Best Programming Practices and Restrictions
## Compiling with AUTO Mode
## PTO AUTO Code Examples 

