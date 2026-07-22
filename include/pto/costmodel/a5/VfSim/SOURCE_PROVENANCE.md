# VfSim Source Provenance

- Upstream: `https://github.com/wang-chonghao/VfSimulator`
- Upstream branch: `master`
- Imported commit: `613f24b6f52e102c7a2e37bb5746cf2685cb88f6`
- Imported on: `2026-07-22`

The native core implementation, typed `VfInfo` API, and `configs/` are vendored
here so the PTO-ISA cost model has no runtime dependency on the upstream
repository. Includes were rewritten from `native/...` and `api/native/...` to
`pto/costmodel/a5/VfSim/...`.

PTO-specific changes are intentionally limited to:

1. `VfSimCostModel.{h,cpp}`, which accepts PTO's `VfInfo` and lowers it into
   upstream typed `vfsim::VfInfo`.
2. Build-system wiring in `cmake/a5_vf_mock.cmake`.
