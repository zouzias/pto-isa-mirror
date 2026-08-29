# 发布说明 — PTO Tile Lib

本文档用于汇总 PTO Tile Lib 的版本变更信息。

格式遵循 Keep a Changelog 风格（Added / Changed / Fixed / Deprecated / Removed / Security）。

## Unreleased

- 暂无

### Added

- PTO Tile Lib 初次公开发布。

### Changed

- 暂无

### Fixed

- CPU_SIM 现已支持带临时 Tile 的 `TCI` 重载，序列行为与不带临时 Tile 的形式一致。
- CPU_SIM `TADD` 现已支持不同的操作数 Tile 类型，包括有效形状兼容的静态/动态类型，同时保留运行时有效形状检查。

### Deprecated

- 暂无

### Removed

- 暂无

### Security

- 漏洞反馈流程请参见 `SECURITY_zh.md`。

## 兼容性说明

- **Ascend（NPU / simulator）**：依赖 Ascend CANN toolkit `>= 8.3`（详见 `version.info`）；具体支持的 SoC 与工具链版本取决于已安装的 CANN 发行版。
- **CPU simulator**：支持在 macOS / Linux / Windows 上结合 C++ 工具链与 Python 运行；详见 `docs/getting-started_zh.md`。
