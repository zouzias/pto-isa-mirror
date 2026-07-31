# Release Notes — PTO Tile Lib

This document summarizes version changes in PTO Tile Lib.

The format follows Keep a Changelog style (Added / Changed / Fixed / Deprecated / Removed / Security).

## Unreleased

- Bump the shared CANN engineering cmake repo tag (`CANN_CMAKE_TAG`) in `cmake/fetch_cann_cmake.cmake` from `master-042` to `master-046`.

### Added

- Initial public release of PTO Tile Lib.

### Changed

- Bump `CANN_CMAKE_TAG` in `cmake/fetch_cann_cmake.cmake` from `master-042` to `master-046` to track the shared CANN engineering cmake repo. The upstream `master-046` release upgrades nlohmann/json to v3.12.0; note that this repo's own `cmake/third_party/json.cmake` already pins json v3.12.0, so no further change is required.

### Fixed

- N/A

### Deprecated

- N/A

### Removed

- N/A

### Security

- For the vulnerability reporting process, see `SECURITY.md`.

## Compatibility Notes

- **Ascend (NPU / simulator)**: requires Ascend CANN toolkit `>= 8.3` (see `version.info`); exact supported SoCs and toolchains depend on the installed CANN distribution.
- **CPU simulator**: intended to run on macOS / Linux / Windows with a C++ toolchain and Python; see `docs/getting-started.md`.
