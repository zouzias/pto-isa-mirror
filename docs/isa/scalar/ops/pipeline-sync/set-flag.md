# pto.set_flag

`pto.set_flag` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Signal an event from one pipeline to another pipeline. This is the producer half of the explicit event protocol that connects MTE, vector, and other execution stages.

## Mechanism

`pto.set_flag` marks the named event as produced by the source pipeline. The signal is visible to matching `wait_flag` operations on the destination pipeline. The event is directional: both the source and destination pipe roles are part of the event identity.

The two pipelines operate asynchronously. The event protocol ensures that the consumer pipeline does not proceed until the producer pipeline has reached a point where the shared data is ready.

## Syntax

### PTO Assembly Form

```text
set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

### AS Level 1 (SSA)

```mlir
pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

## Inputs

| Operand | Type | Description |
|---------|------|-------------|
| `SRC_PIPE` | pipe identifier | Pipeline that produces the event. Common values: `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3` |
| `DST_PIPE` | pipe identifier | Pipeline that is allowed to consume the event |
| `EVENT_ID` | event identifier | Named event slot for this producer-consumer edge |

### Pipe Identifiers

| Pipe | Role |
|------|------|
| `PIPE_MTE2` | DMA engine: GM → UB (load staging) |
| `PIPE_V` | Vector compute pipeline |
| `PIPE_MTE3` | DMA engine: UB → GM (store) |

## Expected Outputs

None. This operation updates pipeline-event state but produces no SSA result.

## Side Effects

Signals the named event. After this operation completes, any `wait_flag` with the matching `(SRC_PIPE, DST_PIPE, EVENT_ID)` tuple becomes eligible to unblock on the destination pipeline.

## Constraints

!!! warning "Constraints"
    - The selected pipe identifiers must be valid for the target profile.
    - The event identifier must be within the supported range for the target profile.
    - The event protocol is directional; using the wrong pipe role is undefined behavior.
    - Portable code must pair each `set_flag` with a corresponding `wait_flag` in the intended dependency chain.

## Common Patterns

### Pattern 1: GM → UB → Compute → UB → GM

This is the standard three-stage pipeline used by vector programs:

```mlir
// Stage 1: MTE2 loads data from GM into UB
pto.copy_gm_to_ubuf %gm_src, %ub_dst, ...

// MTE2 signals: "UB data is ready"
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]

// Stage 2: Vector compute waits, then consumes UB data
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
// ... vector compute on UB data ...

// Vector signals: "UB output is ready"
pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]

// Stage 3: MTE3 waits, then stores result to GM
pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
pto.copy_ubuf_to_gm %ub_src, %gm_dst, ...
```

### Pattern 2: Ping-Pong Double Buffering

When overlapping successive tiles, use separate event IDs for each buffer slot:

```mlir
// Buffer 0: set forward event
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_0"]
// Buffer 1: set forward event
pto.set_flag["PIPE_MTE2", "PIPE_V", "EVT_IN_FWD_1"]
```

## Exceptions

!!! danger "Exceptions"
    - Using invalid pipe identifiers or out-of-range event identifiers is rejected by the verifier.
    - Signaling an event before the producing operation has started produces undefined results.
    - Consuming an event that has not been signaled causes the consumer pipeline to wait indefinitely.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    | Profile | Pipe Support | Event ID Range |
    |---------|-------------|----------------|
    | CPU Simulator | All pipes emulated | Unlimited |
    | A2/A3 | `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3` | Profile-defined |
    | A5 | `PIPE_MTE2`, `PIPE_V`, `PIPE_MTE3` | Profile-defined |

    CPU simulation preserves the event protocol semantics but may not expose all low-level hazards that motivate it on hardware.

## See Also

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Next op in instruction set: [pto.wait_flag](./wait-flag.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
- `pto.wait_flag` for the consumer half of the event protocol
- `pto.pipe_barrier` for draining all pending operations on a pipe
