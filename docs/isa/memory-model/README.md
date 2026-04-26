# Memory Model

This chapter describes PTO's memory consistency model, including visibility and ordering rules, producer-consumer ordering, and the relationship between memory semantics and ISA operations.

## In This Chapter

- [Consistency Baseline](consistency-baseline.md) — memory spaces, ordering classes, and the distinction between undefined, unspecified, and implementation-defined behavior
- [Producer Consumer Ordering](producer-consumer-ordering.md) — state transitions, ordering chains, and cross-instruction-set propagation rules

## Suggested Reading Order

1. Start with [Consistency Baseline](consistency-baseline.md) to understand PTO's memory spaces and ordering vocabulary.
2. Continue with [Producer Consumer Ordering](producer-consumer-ordering.md) to understand how synchronization and data movement establish ordering guarantees.

## Chapter Position

This chapter corresponds to Chapter 6 of the manual. It should be understood before diving into instruction semantics, especially for operations whose behavior depends on ordering and visibility.
