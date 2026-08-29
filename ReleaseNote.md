# Release Notes — PTO Tile Lib

This document summarizes version changes in PTO Tile Lib.

The format follows Keep a Changelog style (Added / Changed / Fixed / Deprecated / Removed / Security).

## Unreleased

- N/A

### Added

- Initial public release of PTO Tile Lib.

### Changed

- N/A

### Fixed

- CPU_SIM now accepts the scratch-tile overload of `TCI` and matches the no-scratch sequence behavior.
- CPU_SIM `TADD` now supports distinct operand Tile types, including compatible static/dynamic valid-shape types,
  while retaining runtime valid-shape checks.

### Deprecated

- N/A

### Removed

- N/A

### Security

- For the vulnerability reporting process, see `SECURITY.md`.

## Compatibility Notes

- **Ascend (NPU / simulator)**: requires Ascend CANN toolkit `>= 8.3` (see `version.info`); exact supported SoCs and toolchains depend on the installed CANN distribution.
- **CPU simulator**: intended to run on macOS / Linux / Windows with a C++ toolchain and Python; see `docs/getting-started.md`.
