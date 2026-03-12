# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
MkDocs build-time generator for PTO Tile Lib.

We intentionally keep MkDocs config under `docs/mkdocs/` and generate a *mirror*
of repository markdown into `docs/mkdocs/src/` so the site can browse markdown
across the entire repo (README files under kernels/, tests/, scripts/, etc.).

Key property:
- Generated pages preserve original repository paths, so existing repo-relative
  links like `docs/...` or `kernels/...` keep working in the site.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mkdocs_gen_files


REPO_ROOT = Path(__file__).resolve().parents[2]

SKIP_PREFIXES = (
    ".git/",
    ".github/",
    ".gitcode/",
    ".venv/",
    ".venv-mkdocs/",
    "site/",
    "site_zh/",
    "build/",
    "build_tests/",
    ".idea/",
    ".vscode/",
)

SKIP_CONTAINS = (
    "/__pycache__/",
    "/CMakeFiles/",
)

ASSET_EXTS = {
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bnf",
}


def _should_skip(rel_posix: str) -> bool:
    if rel_posix.startswith("docs/mkdocs/"):
        return True
    if rel_posix.endswith("/mkdocs.yml"):
        return True
    if rel_posix.startswith(".venv"):
        return True
    if "site-packages/" in rel_posix:
        return True
    if any(rel_posix.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in rel_posix for s in SKIP_CONTAINS):
        return True
    if rel_posix.endswith((".pyc",)):
        return True
    return False


_ABS_LINK_RE = re.compile(r'\]\(/((?!http)[^)]+)\)')
_REL_IMG_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])((?!http|/|data:)[^"\'>]+)(["\'])')


def _rewrite_rel_imgs_for_build(text: str, src_rel: str) -> str:
    """Rewrite relative <img src="..."> paths so they resolve correctly from
    the MkDocs virtual page URL.

    MkDocs serves foo/bar.md at /foo/bar/, so a relative image path that
    works when browsing the repo (relative to foo/) needs to be adjusted
    to be relative to /foo/bar/ instead.

    Example:
      src_rel = "docs/getting-started.md"
      img_path = "figures/pto_logo.svg"  (relative to docs/)
      resolved repo path = docs/figures/pto_logo.svg
      MkDocs page URL   = /docs/getting-started/
      correct rel path  = ../figures/pto_logo.svg
    """
    import posixpath

    # Directory containing the source file (repo-relative, posix).
    src_dir = Path(src_rel).parent.as_posix()  # e.g. "docs"

    # Virtual page directory (where MkDocs serves the page).
    # For foo/bar.md → /foo/bar/  so page_dir = "foo/bar"
    page_dir = Path(src_rel).with_suffix('').as_posix()  # e.g. "docs/getting-started"

    def _replace(m: re.Match) -> str:
        prefix, img_path, suffix = m.group(1), m.group(2), m.group(3)
        # Resolve image to repo-relative path.
        if src_dir and src_dir != '.':
            repo_img = src_dir + '/' + img_path  # e.g. docs/figures/pto_logo.svg
        else:
            repo_img = img_path
        # Normalize (handle any ../ in original img_path).
        repo_img = posixpath.normpath(repo_img)
        # Compute relative path from page_dir to repo_img.
        rel = posixpath.relpath(repo_img, page_dir)  # e.g. ../figures/pto_logo.svg
        return f'{prefix}{rel}{suffix}'

    return _REL_IMG_RE.sub(_replace, text)


def _rewrite_links_for_build(text: str, virtual_path: str) -> str:
    """Rewrite repo-root-absolute links (e.g. /docs/isa/TADD.md) to
    relative links suitable for the MkDocs virtual filesystem.

    Files under docs/mkdocs/src/ are placed at their original relative path
    in the virtual filesystem (e.g. manual/index.md).  A link like
    /docs/isa/TADD.md needs to become ../docs/isa/TADD.md so it resolves
    relative to the virtual file's location.
    """
    # Compute how many path components deep the virtual file is.
    depth = len(Path(virtual_path).parent.parts)
    prefix = '../' * depth if depth else ''

    def _replace(m: re.Match) -> str:
        target = m.group(1)
        return f']({prefix}{target})'

    return _ABS_LINK_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Nav order from mkdocs.yml (used for prev/next generation)
# ---------------------------------------------------------------------------

NAV_PAGES_EN = [
    "index.md",
    "docs/getting-started.md",
    "manual/index.md",
    "manual/01-overview.md",
    "manual/02-machine-model.md",
    "manual/03-state-and-types.md",
    "manual/04-tiles-and-globaltensor.md",
    "manual/05-synchronization.md",
    "manual/06-assembly.md",
    "manual/07-instructions.md",
    "manual/08-programming.md",
    "manual/09-virtual-isa-and-ir.md",
    "manual/10-bytecode-and-toolchain.md",
    "manual/11-memory-ordering-and-consistency.md",
    "manual/12-backend-profiles-and-conformance.md",
    "manual/appendix-a-glossary.md",
    "manual/appendix-b-instruction-contract-template.md",
    "manual/appendix-c-diagnostics-taxonomy.md",
    "manual/appendix-d-instruction-family-matrix.md",
    "docs/coding/README.md",
    "docs/coding/ProgrammingModel.md",
    "docs/coding/Tile.md",
    "docs/coding/GlobalTensor.md",
    "docs/coding/Scalar.md",
    "docs/coding/Event.md",
    "docs/coding/tutorial.md",
    "docs/coding/tutorials/README.md",
    "docs/coding/tutorials/vec-add.md",
    "docs/coding/tutorials/row-softmax.md",
    "docs/coding/tutorials/gemm.md",
    "docs/coding/opt.md",
    "docs/coding/debug.md",
    "docs/machine/abstract-machine.md",
    "docs/machine/README.md",
    "docs/isa/README.md",
    "docs/isa/conventions.md",
    "docs/assembly/README.md",
    "docs/assembly/PTO-AS.md",
    "docs/assembly/conventions.md",
    "docs/assembly/nonisa-ops.md",
    "docs/assembly/elementwise-ops.md",
    "docs/assembly/tile-scalar-ops.md",
    "docs/assembly/axis-ops.md",
    "docs/assembly/memory-ops.md",
    "docs/assembly/matrix-ops.md",
    "docs/assembly/data-movement-ops.md",
    "docs/assembly/complex-ops.md",
    "docs/assembly/manual-binding-ops.md",
    "docs/assembly/scalar-arith-ops.md",
    "docs/assembly/control-flow-ops.md",
    "docs/PTOISA.md",
    "docs/isa/TSYNC.md",
    "docs/isa/TASSIGN.md",
    "docs/isa/TSETFMATRIX.md",
    "docs/isa/TSET_IMG2COL_RPT.md",
    "docs/isa/TSET_IMG2COL_PADDING.md",
    "docs/isa/TADD.md", "docs/isa/TABS.md", "docs/isa/TAND.md",
    "docs/isa/TOR.md", "docs/isa/TSUB.md", "docs/isa/TMUL.md",
    "docs/isa/TMIN.md", "docs/isa/TMAX.md", "docs/isa/TCMP.md",
    "docs/isa/TDIV.md", "docs/isa/TSHL.md", "docs/isa/TSHR.md",
    "docs/isa/TXOR.md", "docs/isa/TLOG.md", "docs/isa/TRECIP.md",
    "docs/isa/TPRELU.md", "docs/isa/TADDC.md", "docs/isa/TSUBC.md",
    "docs/isa/TCVT.md", "docs/isa/TSEL.md", "docs/isa/TRSQRT.md",
    "docs/isa/TSQRT.md", "docs/isa/TEXP.md", "docs/isa/TNOT.md",
    "docs/isa/TRELU.md", "docs/isa/TNEG.md", "docs/isa/TREM.md",
    "docs/isa/TFMOD.md",
    "docs/isa/TEXPANDS.md", "docs/isa/TCMPS.md", "docs/isa/TSELS.md",
    "docs/isa/TMINS.md", "docs/isa/TADDS.md", "docs/isa/TSUBS.md",
    "docs/isa/TDIVS.md", "docs/isa/TMULS.md", "docs/isa/TFMODS.md",
    "docs/isa/TREMS.md", "docs/isa/TMAXS.md", "docs/isa/TANDS.md",
    "docs/isa/TORS.md", "docs/isa/TSHLS.md", "docs/isa/TSHRS.md",
    "docs/isa/TXORS.md", "docs/isa/TLRELU.md", "docs/isa/TADDSC.md",
    "docs/isa/TSUBSC.md",
    "docs/isa/TROWSUM.md", "docs/isa/TROWPROD.md", "docs/isa/TCOLSUM.md",
    "docs/isa/TCOLPROD.md", "docs/isa/TCOLMAX.md", "docs/isa/TROWMAX.md",
    "docs/isa/TROWMIN.md", "docs/isa/TCOLMIN.md", "docs/isa/TROWEXPAND.md",
    "docs/isa/TROWEXPANDDIV.md", "docs/isa/TROWEXPANDMUL.md",
    "docs/isa/TROWEXPANDSUB.md", "docs/isa/TROWEXPANDADD.md",
    "docs/isa/TROWEXPANDMAX.md", "docs/isa/TROWEXPANDMIN.md",
    "docs/isa/TROWEXPANDEXPDIF.md", "docs/isa/TCOLEXPAND.md",
    "docs/isa/TCOLEXPANDDIV.md", "docs/isa/TCOLEXPANDMUL.md",
    "docs/isa/TCOLEXPANDADD.md", "docs/isa/TCOLEXPANDMAX.md",
    "docs/isa/TCOLEXPANDMIN.md", "docs/isa/TCOLEXPANDSUB.md",
    "docs/isa/TCOLEXPANDEXPDIF.md",
    "docs/isa/TLOAD.md", "docs/isa/TPREFETCH.md", "docs/isa/TSTORE.md",
    "docs/isa/TSTORE_FP.md", "docs/isa/MGATHER.md", "docs/isa/MSCATTER.md",
    "docs/isa/TMATMUL.md", "docs/isa/TMATMUL_ACC.md", "docs/isa/TMATMUL_BIAS.md",
    "docs/isa/TMATMUL_MX.md", "docs/isa/TGEMV.md", "docs/isa/TGEMV_ACC.md",
    "docs/isa/TGEMV_BIAS.md", "docs/isa/TGEMV_MX.md",
    "docs/isa/TMOV.md", "docs/isa/TMOV_FP.md", "docs/isa/TEXTRACT.md",
    "docs/isa/TEXTRACT_FP.md", "docs/isa/TINSERT.md", "docs/isa/TINSERT_FP.md",
    "docs/isa/TFILLPAD.md", "docs/isa/TFILLPAD_INPLACE.md",
    "docs/isa/TFILLPAD_EXPAND.md", "docs/isa/TRESHAPE.md",
    "docs/isa/TTRANS.md", "docs/isa/TIMG2COL.md",
    "docs/isa/TGATHER.md", "docs/isa/TGATHERB.md", "docs/isa/TSCATTER.md",
    "docs/isa/TCI.md", "docs/isa/TTRI.md", "docs/isa/TPARTADD.md",
    "docs/isa/TPARTMUL.md", "docs/isa/TPARTMAX.md", "docs/isa/TPARTMIN.md",
    "docs/isa/TSORT32.md", "docs/isa/TMRGSORT.md", "docs/isa/TQUANT.md",
    "docs/isa/TPRINT.md",
    "docs/reference/pto-intrinsics-header.md",
    "manual/isa-reference.md",
    # Examples & Kernels
    "kernels/README.md",
    "kernels/manual/a2a3/gemm_performance/README.md",
    "kernels/manual/common/flash_atten/README.md",
    "demos/baseline/add/README.md",
    "demos/baseline/gemm_basic/README.md",
    "tests/README.md",
    "tests/script/README.md",
    # Documentation
    "docs/README.md",
    "docs/website.md",
]


def _md_to_url(md_path: str) -> str:
    """Convert a virtual .md path to the MkDocs site URL path.

    MkDocs converts:
      - ``foo/index.md``  → ``/foo/``
      - ``foo/README.md`` → ``/foo/``   (README treated as directory index)
      - ``index.md``      → ``/``
      - ``README.md``     → ``/``
      - ``foo/bar.md``    → ``/foo/bar/``
    """
    p = Path(md_path)
    if p.name in ("index.md", "README.md"):
        parent = p.parent.as_posix().lstrip("./")
        url = "/" + parent + "/" if parent else "/"
    else:
        url = "/" + p.with_suffix("").as_posix().lstrip("./") + "/"
    # normalise double-slash at root
    if url == "//":
        url = "/"
    return url


def _en_url_to_zh_url(en_url: str) -> str | None:
    """Best-effort mapping: English URL → Chinese URL.

    Returns None if we cannot determine the zh counterpart.
    """
    # root index: / -> /index_zh/
    if en_url == "/":
        return "/index_zh/"
    # strip trailing slash for manipulation
    base = en_url.rstrip("/")
    # manual index: /manual -> /manual/index_zh
    if base == "/manual":
        return "/manual/index_zh/"
    # README pages: last segment is a known directory name
    README_DIRS = {
        "coding", "isa", "machine", "assembly", "docs", "kernels",
        "tests", "demos", "scripts", "include", "cmake", "reference",
        "tutorials", "script", "package", "custom", "baseline", "add",
        "gemm_basic", "flash_atten", "gemm_performance", "a2a3", "a5",
        "kirin9030", "npu", "pto",
    }
    last = base.rsplit("/", 1)[-1]
    if last in README_DIRS:
        return en_url.rstrip("/") + "/README_zh/"
    # general page: append _zh
    return base + "_zh/"


def generate_lang_map(nav_pages: list[str]) -> dict:
    """Build a mapping dict for use by the language switcher JS.

    Structure::

        {
          "en_to_zh": { "/manual/01-overview/": "/manual/01-overview_zh/", ... },
          "zh_to_en": { "/manual/01-overview_zh/": "/manual/01-overview/", ... },
          "nav": [
            { "en": "/manual/01-overview/", "zh": "/manual/01-overview_zh/",
              "prev_en": "/manual/", "prev_zh": "/manual/index_zh/",
              "next_en": "/manual/02-machine-model/",
              "next_zh": "/manual/02-machine-model_zh/" },
            ...
          ]
        }
    """
    en_urls = [_md_to_url(p) for p in nav_pages]
    en_to_zh: dict[str, str] = {}
    zh_to_en: dict[str, str] = {}

    for en in en_urls:
        zh = _en_url_to_zh_url(en)
        if zh:
            en_to_zh[en] = zh
            zh_to_en[zh] = en

    nav_entries = []
    for i, en in enumerate(en_urls):
        zh = en_to_zh.get(en)
        prev_en = en_urls[i - 1] if i > 0 else None
        next_en = en_urls[i + 1] if i < len(en_urls) - 1 else None
        entry = {
            "en": en,
            "zh": zh,
            "prev_en": prev_en,
            "prev_zh": en_to_zh.get(prev_en) if prev_en else None,
            "next_en": next_en,
            "next_zh": en_to_zh.get(next_en) if next_en else None,
        }
        nav_entries.append(entry)

    return {"en_to_zh": en_to_zh, "zh_to_en": zh_to_en, "nav": nav_entries}


def main() -> None:
    copied_md: list[str] = []

    # Mirror markdown files into the MkDocs virtual filesystem, preserving paths.
    mkdocs_src = REPO_ROOT / "docs" / "mkdocs" / "src"
    for src in REPO_ROOT.rglob("*.md"):
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        # Use utf-8-sig to automatically remove BOM if present
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        # Rewrite relative <img src="..."> paths for all mirrored files.
        text = _rewrite_rel_imgs_for_build(text, rel)
        # For hand-written files under docs/mkdocs/src/, rewrite repo-root-absolute
        # links (e.g. /docs/isa/TADD.md) to relative paths for MkDocs.
        # These files use absolute-style links so they resolve correctly when
        # browsing the repository statically (GitHub/Gitee), and this step
        # converts them to the relative paths that MkDocs expects at build time.
        try:
            src.relative_to(mkdocs_src)
            _is_under_mkdocs_src = True
        except ValueError:
            _is_under_mkdocs_src = False
        if _is_under_mkdocs_src:
            virtual_path = src.relative_to(mkdocs_src).as_posix()
            text = _rewrite_links_for_build(text, virtual_path)
        with mkdocs_gen_files.open(rel, "w") as f:
            f.write(f"<!-- Generated from `{rel}` -->\n\n")
            f.write(text)
        copied_md.append(rel)

    # Generate per-instruction reference indexes for docs/isa/*.md.
    isa_dir = REPO_ROOT / "docs" / "isa"
    isa_pages_en: list[tuple[str, str]] = []
    isa_pages_zh: list[tuple[str, str]] = []

    def extract_first_heading(md_path: Path) -> str:
        try:
            # Use utf-8-sig to automatically remove BOM if present
            text = md_path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return md_path.stem
        for line in text.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return md_path.stem

    if isa_dir.exists():
        for p in sorted(isa_dir.glob("*.md")):
            if p.name in ("README.md", "README_zh.md", "conventions.md", "conventions_zh.md"):
                continue
            stem = p.stem
            title = extract_first_heading(p)
            if stem.endswith("_zh"):
                isa_pages_zh.append((stem, title))
            else:
                isa_pages_en.append((stem, title))

    with mkdocs_gen_files.open("manual/isa-reference.md", "w") as f:
        f.write("# Instruction Reference Pages\n\n")
        f.write("This page is generated at build time.\n\n")
        f.write("- Instruction index: `docs/isa/README.md`\n")
        f.write("- ISA conventions: `docs/isa/conventions.md`\n\n")
        if not isa_pages_en:
            f.write("No English instruction pages were found under `docs/isa/`.\n")
        else:
            f.write("## All instructions\n\n")
            for instr, title in isa_pages_en:
                # This page lives under `manual/`, so use `../` to link back to root-level docs.
                link = f"../docs/isa/{instr}.md"
                suffix = "" if title.strip() == instr else f" — {title}"
                f.write(f"- [{instr}]({link}){suffix}\n")
            f.write("\n")

    with mkdocs_gen_files.open("manual/isa-reference_zh.md", "w") as f:
        f.write("# 指令参考页面（全量）\n\n")
        f.write("本页在构建站点时自动生成。\n\n")
        f.write("- 指令索引：`docs/isa/README_zh.md`\n")
        f.write("- ISA 通用约定：`docs/isa/conventions_zh.md`\n\n")
        if not isa_pages_zh:
            f.write("未在 `docs/isa/` 下发现中文指令页面。\n")
        else:
            f.write("## 全部指令\n\n")
            for instr, title in isa_pages_zh:
                link = f"../docs/isa/{instr}.md"
                suffix = "" if title.strip() == instr else f" — {title}"
                f.write(f"- [{instr}]({link}){suffix}\n")
            f.write("\n")

    # Generate a simple index page that links to all mirrored markdown.
    copied_md = sorted(set(copied_md))
    sections: dict[str, list[str]] = {}
    sections_zh: dict[str, list[str]] = {}
    
    for rel in copied_md:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        sections.setdefault(top, []).append(rel)
        
        # 同时为中文版分类
        if "_zh.md" in rel or rel.endswith("_zh/index.md"):
            sections_zh.setdefault(top, []).append(rel)

    # Generate English all-pages.md
    with mkdocs_gen_files.open("all-pages.md", "w") as f:
        f.write("# All Markdown Pages\n\n")
        f.write("This page is generated at build time and lists markdown files mirrored into the site.\n\n")
        for top in sorted(sections.keys()):
            f.write(f"## {top}\n\n")
            for rel in sections[top]:
                # Prefer a short label but keep it unambiguous.
                label = rel if top == "(root)" else rel[len(top) + 1 :]
                f.write(f"- [{label}]({rel})\n")
            f.write("\n")
    
    # Generate Chinese all-pages_zh.md
    with mkdocs_gen_files.open("all-pages_zh.md", "w") as f:
        f.write("# 所有 Markdown 页面\n\n")
        f.write("本页面在构建时自动生成，列出了站点中镜像的所有中文 markdown 文件。\n\n")
        if not sections_zh:
            f.write("未找到中文页面。\n")
        else:
            for top in sorted(sections_zh.keys()):
                f.write(f"## {top}\n\n")
                for rel in sections_zh[top]:
                    # Prefer a short label but keep it unambiguous.
                    label = rel if top == "(root)" else rel[len(top) + 1 :]
                    f.write(f"- [{label}]({rel})\n")
                f.write("\n")

    # Generate lang-map.json for zero-latency language switching.
    lang_map = generate_lang_map(NAV_PAGES_EN)
    with mkdocs_gen_files.open("lang-map.json", "w") as f:
        json.dump(lang_map, f, ensure_ascii=False, separators=(",", ":"))

    # Mirror commonly referenced doc assets (images) so docs render cleanly.
    for src in REPO_ROOT.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in ASSET_EXTS:
            continue
        rel = src.relative_to(REPO_ROOT).as_posix()
        if _should_skip(rel):
            continue
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(src.read_bytes())


main()
