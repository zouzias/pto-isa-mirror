# moe_dispatch — A2/A3 PTO MoE Dispatch Kernel

Hand-written PTO kernel for the **MoE dispatch** stage on Ascend A2/A3.

```text
inputA[local M, K] + expertIdx[local M, topK]
  -> pack local rows by global expert into peerWindow.packedA
  -> publish per-expert count rows to peer windows
  -> wait for all peer count rows
  -> build local expert prefix metadata
  -> gather owner expert payload with TGET into workspace.dispatchedA
```

## Scope

Included:

- Routing count and local pack.
- Cross-rank count publication with PTO communication primitives.
- Prefix metadata for local expert ownership.
- Owner-side payload gather into `workspace.dispatchedA`.
- CPU golden metadata and payload verification.

Not included:

- Expert FFN/GMM compute.
- Combine return / weighted restore.
- PTO style cleanup beyond preserving the existing working dispatch path.

## Build And Run

Build only:

```bash
bash run.sh --skip-run 1 --clean-build 1
```

Small verification run:

```bash
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2
```

Default verification compares dispatch metadata and payload against the CPU golden reference. Expected markers:

```text
dispatch_metadata_mismatches=0
dispatch_payload_mismatches=0
```

## Main Files

- `moe_dispatch_kernel.cpp`: device dispatch kernel and launcher.
- `main.cpp`: MPI/HCCL host orchestration and dispatch verification.
- `common.h`: shared host/device ABI structs.
- `layout.h`: host layout calculators matching device layout.
- `golden.h`: deterministic input generation and CPU reference dispatch.
