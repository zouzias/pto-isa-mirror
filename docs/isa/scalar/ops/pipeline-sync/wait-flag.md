# pto.wait_flag

`pto.wait_flag` is part of the [Pipeline Sync](../../pipeline-sync.md) instruction set.

## Summary

Block the destination pipeline until a matching event is signaled.

## Mechanism

`pto.wait_flag` is the consumer half of the explicit event protocol. The destination pipeline stalls until the matching `(SRC_PIPE, DST_PIPE, EVENT_ID)` signal becomes visible, after which later work on the destination pipeline may proceed.

## Syntax

### PTO Assembly Form

```text
wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

### AS Level 1 (SSA)

```mlir
pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

## Inputs

| Operand | Type | Description |
| --- | --- | --- |
| `SRC_PIPE` | pipe identifier | Pipeline expected to produce the event |
| `DST_PIPE` | pipe identifier | Pipeline that must wait for the event |
| `EVENT_ID` | event identifier | Named event slot that forms the producer-consumer edge |

## Expected Outputs

| Result | Type | Description |
| --- | --- | --- |
| None | `—` | This form does not return SSA values; it blocks pipeline progress until the event becomes visible. |

## Side Effects

Stalls the destination pipeline until the matching event is signaled. Other pipelines may continue according to the target profile.

## Constraints

!!! warning "Constraints"
    - The selected pipe identifiers and event identifier MUST be valid for the selected target profile.
    - Waiting on an event that is never signaled is an illegal PTO program.
    - Portable code MUST pair each wait with the intended producer-side `pto.set_flag`.

## Exceptions

!!! danger "Exceptions"
    - The verifier rejects illegal operand shapes, unsupported pipe or event identifiers, and attribute combinations that are not valid for the selected instruction set or target profile.
    - Any additional illegality stated in the constraints section is also part of the contract.

## Target-Profile Restrictions

??? info "Target-Profile Restrictions"
    - CPU simulation preserves the visible event protocol but may not expose all low-level hazards that motivate it on hardware.
    - A2/A3 and A5 may use different concrete pipe and event spaces; portable code must rely on the documented PTO contract plus the selected target profile.

## Examples

```mlir
pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]
```

## Related Ops / Instruction Set Links

- Instruction set overview: [Pipeline Sync](../../pipeline-sync.md)
- Previous op in instruction set: [pto.set_flag](./set-flag.md)
- Next op in instruction set: [pto.pipe_barrier](./pipe-barrier.md)
- Control-shell overview: [Control and configuration](../../control-and-configuration.md)
