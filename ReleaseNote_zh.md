# 发布说明 — PTO Tile Lib

本文档用于汇总 PTO Tile Lib 的版本变更信息。

格式遵循 Keep a Changelog 风格（Added / Changed / Fixed / Deprecated / Removed / Security）。

## Unreleased

- 将 `cmake/fetch_cann_cmake.cmake` 中的 CANN 工程公共 cmake 仓 tag（`CANN_CMAKE_TAG`）从 `master-042` 升级到 `master-046`。

### Added

- PTO Tile Lib 初次公开发布。

### Changed

- 将 `cmake/fetch_cann_cmake.cmake` 中的 `CANN_CMAKE_TAG` 从 `master-042` 升级到 `master-046`，跟随 CANN 工程公共 cmake 仓。上游 `master-046` 将 nlohmann/json 升级到 v3.12.0；本仓自有的 `cmake/third_party/json.cmake` 已固定为 json v3.12.0，无需额外改动。

### Fixed

- 暂无

### Deprecated

- 暂无

### Removed

- 暂无

### Security

- 漏洞反馈流程请参见 `SECURITY_zh.md`。

## 兼容性说明

- **Ascend（NPU / simulator）**：依赖 Ascend CANN toolkit `>= 8.3`（详见 `version.info`）；具体支持的 SoC 与工具链版本取决于已安装的 CANN 发行版。
- **CPU simulator**：支持在 macOS / Linux / Windows 上结合 C++ 工具链与 Python 运行；详见 `docs/getting-started_zh.md`。
