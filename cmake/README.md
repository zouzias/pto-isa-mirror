# CMake/

This directory contains CMake helper modules used by the repository, mainly for packaging and third-party dependency integration.

## What’s Inside

- Packaging-related CMake logic (the `package` target)
- FetchContent integration that pulls the shared CANN engineering CMake repo
- Third-party dependency download/integration scripts under `CMake/third_party/`

## Key Files

- `CMake/package.CMake`: Packaging entry functions included by the top-level `CMakeLists.txt`; delegates the CPack configuration to `set_cann_cpack_config` provided by the engineering repo
- `CMake/fetch_cann_cmake.CMake`: Fetches `https://gitcode.com/cann/CMake` and exposes its common functions (`init_cann_project`, `set_cann_cpack_config`, the bundled `makeself`/install scripts, etc.)
- `CMake/func.CMake`: Project-local helpers (protobuf, signing, packing)
- `CMake/third_party/`: Third-party dependency helper scripts

## Entry Points

- Top-level `CMakeLists.txt` includes `CMake/package.CMake` and invokes the packaging helpers
- `build.sh --pkg` triggers the repository packaging flow
