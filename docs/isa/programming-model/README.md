# Programming Model

This chapter describes the objects programmers reason about and manipulate in PTO: Tiles, valid regions, GlobalTensor views, and the execution models of Auto mode and Manual mode.

## In This Chapter

- [Tiles And Valid Regions](tiles-and-valid-regions.md) — tile roles, valid regions, and their constraints
- [GlobalTensor And Data Movement](globaltensor-and-data-movement.md) — GlobalTensor views and movement between GM and Tile storage
- [Auto Vs Manual](auto-vs-manual.md) — comparison of the two execution modes and when to use each

## Suggested Reading Order

1. Start with [Tiles And Valid Regions](tiles-and-valid-regions.md) to understand PTO's core abstraction.
2. Continue with [GlobalTensor And Data Movement](globaltensor-and-data-movement.md) to understand how data moves into and out of tiles.
3. Finish with [Auto Vs Manual](auto-vs-manual.md) to choose the right execution mode for a workload.

## Chapter Position

This chapter corresponds to Chapter 2 of the manual. It should be read before instruction-level details because PTO instructions are defined around tiles and valid regions.
