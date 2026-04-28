# docs/mkdocs Documentation Build Guide

This directory is used to build the online documentation and local static documentation site for PTO Tile Lib, based on MkDocs.

## Documentation Content

The generated documentation covers:

- PTO ISA instruction reference
- PTO assembly syntax and specification (PTO-AS)
- Programming model and developer documentation
- Getting started and usage guides
- Kernel examples and directory guides

The authored documentation sources are mainly located in the main repo trees:

- `docs/isa/` for the PTO ISA manual and grouped instruction reference
- `docs/assembly/` for PTO-AS syntax/spec material
- top-level manual entry pages such as `docs/PTO-Virtual-ISA-Manual.md`

`docs/mkdocs/src/` only contains MkDocs-owned landing pages, theme assets, and
build-time generated virtual pages. In particular, `docs/mkdocs/src/docs/**`
should be treated as generated output from `gen_pages.py`, not as a hand-edited
source tree.

## Recommended Usage

- Read documentation online: visit the [Documentation Center](https://pto-isa.gitcode.com)
- Preview locally or browse offline: build with MkDocs locally

## Prerequisites

- Python >= 3.8
- pip

It is recommended to create a dedicated Python virtual environment first.

## Option 1: Use MkDocs CLI

### 1. Install dependencies

```bash
python -m pip install -r docs/mkdocs/requirements.txt
```

### 2. Preview locally

```bash
python -m mkdocs serve -f docs/mkdocs/mkdocs.yml
```

After startup, the documentation is available at `http://127.0.0.1:8000`, and local changes are hot-reloaded automatically.

### 3. Build a static site

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml
```

The output is generated in the repo-level `site/` directory.

## Option 2: Build via CMake

This is suitable for integrating documentation builds into development workflows or CI/CD.

### 1. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv-mkdocs
source .venv-mkdocs/bin/activate  # Windows: .venv-mkdocs\Scripts\Activate.ps1
python -m pip install -r docs/mkdocs/requirements.txt
```

### 2. Configure and build

```bash
cmake -S docs -B build/docs -DPython3_EXECUTABLE=$PWD/.venv-mkdocs/bin/python
cmake --build build/docs --target pto_docs
```

Windows (PowerShell):

```powershell
cmake -S docs -B build/docs -DPython3_EXECUTABLE="$PWD\.venv-mkdocs\Scripts\python.exe"
cmake --build build/docs --target pto_docs
```

The build output is located in `build/docs/site/`.

## Directory Overview

- `mkdocs.yml`: MkDocs configuration file
- `requirements.txt`: documentation build dependencies
- `src/`: MkDocs-owned landing pages and assets
- `gen_pages.py`: documentation page generation script
- `check_mkdocs.py`: documentation build check script

## Related Documents

- [Root README](../../README.md)
- [Getting Started Guide](../getting-started.md)
- [Documentation Entry](../README.md)
