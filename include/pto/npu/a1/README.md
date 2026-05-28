# include/pto/npu/a1/

Ascend A1 series PTO instruction implementation headers.

## Overview

- Implementations are organized per instruction (or instruction family), for example: `TAdd.hpp`, `TMatmul.hpp`, `TLoad.hpp`, `TStore.hpp`
- Some shared operator patterns are also provided (for example, Reduce/Expand/PartOp helpers)

## Related

- ISA semantics and examples: `docs/isa/`
- A1 NPU ST tests: `tests/npu/a1/src/st/`