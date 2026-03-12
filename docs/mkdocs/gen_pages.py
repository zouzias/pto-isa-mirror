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


def shouldSkip(relPosix: str) -> bool:
    if relPosix.startswith("docs/mkdocs/"):
        return True
    if relPosix.endswith("/mkdocs.yml"):
        return True
    if relPosix.startswith(".venv"):
        return True
    if "site-packages/" in relPosix:
        return True
    if any(relPosix.startswith(p) for p in SKIP_PREFIXES):
        return True
    if any(s in relPosix for s in SKIP_CONTAINS):
        return True
    if relPosix.endswith((".pyc",)):
        return True
    return False


ABS_LINK_RE = re.compile(r'\]\(/((?!http)[^)]+)\)')
REL_IMG_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])((?!http|/|data:)[^"\'>]+)(["\'])')


def rewriteRelImgsForBuild(text: str, srcRel: str) -> str:
    """Rewrite relative <img src="..."> paths so they resolve correctly from
    the MkDocs virtual page URL.

    MkDocs serves foo/bar.md at /foo/bar/, so a relative image path that
    works when browsing the repo (relative to foo/) needs to be adjusted
    to be relative to /foo/bar/ instead.

    Example:
      srcRel = "docs/getting-started.md"
      imgPath = "figures/pto_logo.svg"  (relative to docs/)
      resolved repo path = docs/figures/pto_logo.svg
      MkDocs page URL   = /docs/getting-started/
      correct rel path  = ../figures/pto_logo.svg
    """
    import posixpath

    # Directory containing the source file (repo-relative, posix).
    srcDir = Path(srcRel).parent.as_posix()  # e.g. "docs"

    # Virtual page directory (where MkDocs serves the page).
    # For foo/bar.md → /foo/bar/  so pageDir = "foo/bar"
    pageDir = Path(srcRel).with_suffix('').as_posix()  # e.g. "docs/getting-started"

    def replace(m: re.Match) -> str:
        prefix, imgPath, suffix = m.group(1), m.group(2), m.group(3)
        # Resolve image to repo-relative path.
        if srcDir and srcDir != '.':
            repoImg = srcDir + '/' + imgPath  # e.g. docs/figures/pto_logo.svg
        else:
            repoImg = imgPath
        # Normalize (handle any ../ in original imgPath).
        repoImg = posixpath.normpath(repoImg)
        # Compute relative path from pageDir to repoImg.
        rel = posixpath.relpath(repoImg, pageDir)  # e.g. ../figures/pto_logo.svg
        return f'{prefix}{rel}{suffix}'

    return REL_IMG_RE.sub(replace, text)


def rewriteLinksForBuild(text: str, virtualPath: str) -> str:
    """Rewrite repo-root-absolute links (e.g. /docs/isa/TADD.md) to
    relative links suitable for the MkDocs virtual filesystem.

    Files under docs/mkdocs/src/ are placed at their original relative path
    in the virtual filesystem (e.g. manual/index.md).  A link like
    /docs/isa/TADD.md needs to become ../docs/isa/TADD.md so it resolves
    relative to the virtual file's location.
    """
    # Compute how many path components deep the virtual file is.
    depth = len(Path(virtualPath).parent.parts)
    prefix = '../' * depth if depth else ''

    def replace(m: re.Match) -> str:
        target = m.group(1)
        return f']({prefix}{target})'

    return ABS_LINK_RE.sub(replace, text)


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


def mdToUrl(mdPath: str) -> str:
    """Convert a virtual .md path to the MkDocs site URL path.

    MkDocs converts:
      - ``foo/index.md``  → ``/foo/``
      - ``foo/README.md`` → ``/foo/``   (README treated as directory index)
      - ``index.md``      → ``/``
      - ``README.md``     → ``/``
      - ``foo/bar.md``    → ``/foo/bar/``
    """
    p = Path(mdPath)
    if p.name in ("index.md", "README.md"):
        parent = p.parent.as_posix().lstrip("./")
        url = "/" + parent + "/" if parent else "/"
    else:
        url = "/" + p.with_suffix("").as_posix().lstrip("./") + "/"
    # normalise double-slash at root
    if url == "//":
        url = "/"
    return url


def enUrlToZhUrl(enUrl: str) -> str | None:
    """Best-effort mapping: English URL → Chinese URL.

    Returns None if we cannot determine the zh counterpart.
    """
    # root index: / -> /index_zh/
    if enUrl == "/":
        return "/index_zh/"
    # strip trailing slash for manipulation
    base = enUrl.rstrip("/")
    # manual index: /manual -> /manual/index_zh
    if base == "/manual":
        return "/manual/index_zh/"
    # README pages: last segment is a known directory name
    readmeDirs = {
        "coding", "isa", "machine", "assembly", "docs", "kernels",
        "tests", "demos", "scripts", "include", "cmake", "reference",
        "tutorials", "script", "package", "custom", "baseline", "add",
        "gemm_basic", "flash_atten", "gemm_performance", "a2a3", "a5",
        "kirin9030", "npu", "pto",
    }
    last = base.rsplit("/", 1)[-1]
    if last in readmeDirs:
        return enUrl.rstrip("/") + "/README_zh/"
    # general page: append _zh
    return base + "_zh/"


def generateLangMap(navPages: list[str]) -> dict:
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
    enUrls = [mdToUrl(p) for p in navPages]
    enToZh: dict[str, str] = {}
    zhToEn: dict[str, str] = {}

    for en in enUrls:
        zh = enUrlToZhUrl(en)
        if zh:
            enToZh[en] = zh
            zhToEn[zh] = en

    navEntries = []
    for i, en in enumerate(enUrls):
        zh = enToZh.get(en)
        prevEn = enUrls[i - 1] if i > 0 else None
        nextEn = enUrls[i + 1] if i < len(enUrls) - 1 else None
        entry = {
            "en": en,
            "zh": zh,
            "prev_en": prevEn,
            "prev_zh": enToZh.get(prevEn) if prevEn else None,
            "next_en": nextEn,
            "next_zh": enToZh.get(nextEn) if nextEn else None,
        }
        navEntries.append(entry)

    return {"en_to_zh": enToZh, "zh_to_en": zhToEn, "nav": navEntries}


def main() -> None:
    copiedMd: list[str] = []

    # Mirror markdown files into the MkDocs virtual filesystem, preserving paths.
    mkdocsSrc = REPO_ROOT / "docs" / "mkdocs" / "src"
    for src in REPO_ROOT.rglob("*.md"):
        rel = src.relative_to(REPO_ROOT).as_posix()
        if shouldSkip(rel):
            continue
        # Use utf-8-sig to automatically remove BOM if present
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        # Rewrite relative <img src="..."> paths for all mirrored files.
        text = rewriteRelImgsForBuild(text, rel)
        # For hand-written files under docs/mkdocs/src/, rewrite repo-root-absolute
        # links (e.g. /docs/isa/TADD.md) to relative paths for MkDocs.
        # These files use absolute-style links so they resolve correctly when
        # browsing the repository statically (GitHub/Gitee), and this step
        # converts them to the relative paths that MkDocs expects at build time.
        try:
            src.relative_to(mkdocsSrc)
            isUnderMkdocsSrc = True
        except ValueError:
            isUnderMkdocsSrc = False
        if isUnderMkdocsSrc:
            virtualPath = src.relative_to(mkdocsSrc).as_posix()
            text = rewriteLinksForBuild(text, virtualPath)
        with mkdocs_gen_files.open(rel, "w") as f:
            f.write(f"<!-- Generated from `{rel}` -->\n\n")
            f.write(text)
        copiedMd.append(rel)

    # Generate per-instruction reference indexes for docs/isa/*.md.
    isaDir = REPO_ROOT / "docs" / "isa"
    isaPageEn: list[tuple[str, str]] = []
    isaPageZh: list[tuple[str, str]] = []

    def extractFirstHeading(mdPath: Path) -> str:
        try:
            # Use utf-8-sig to automatically remove BOM if present
            text = mdPath.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return mdPath.stem
        for line in text.splitlines():
            if line.startswith("#"):
                return line.lstrip("#").strip()
        return mdPath.stem

    if isaDir.exists():
        for p in sorted(isaDir.glob("*.md")):
            if p.name in ("README.md", "README_zh.md", "conventions.md", "conventions_zh.md"):
                continue
            stem = p.stem
            title = extractFirstHeading(p)
            if stem.endswith("_zh"):
                isaPageZh.append((stem, title))
            else:
                isaPageEn.append((stem, title))

    with mkdocs_gen_files.open("manual/isa-reference.md", "w") as f:
        f.write("# Instruction Reference Pages\n\n")
        f.write("This page is generated at build time.\n\n")
        f.write("- Instruction index: `docs/isa/README.md`\n")
        f.write("- ISA conventions: `docs/isa/conventions.md`\n\n")
        if not isaPageEn:
            f.write("No English instruction pages were found under `docs/isa/`.\n")
        else:
            f.write("## All instructions\n\n")
            for instr, title in isaPageEn:
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
        if not isaPageZh:
            f.write("未在 `docs/isa/` 下发现中文指令页面。\n")
        else:
            f.write("## 全部指令\n\n")
            for instr, title in isaPageZh:
                link = f"../docs/isa/{instr}.md"
                suffix = "" if title.strip() == instr else f" — {title}"
                f.write(f"- [{instr}]({link}){suffix}\n")
            f.write("\n")

    # Generate a simple index page that links to all mirrored markdown.
    copiedMd = sorted(set(copiedMd))
    sections: dict[str, list[str]] = {}
    sectionsZh: dict[str, list[str]] = {}
    
    for rel in copiedMd:
        top = rel.split("/", 1)[0] if "/" in rel else "(root)"
        sections.setdefault(top, []).append(rel)
        
        # 同时为中文版分类
        if "_zh.md" in rel or rel.endswith("_zh/index.md"):
            sectionsZh.setdefault(top, []).append(rel)

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
        if not sectionsZh:
            f.write("未找到中文页面。\n")
        else:
            for top in sorted(sectionsZh.keys()):
                f.write(f"## {top}\n\n")
                for rel in sectionsZh[top]:
                    # Prefer a short label but keep it unambiguous.
                    label = rel if top == "(root)" else rel[len(top) + 1 :]
                    f.write(f"- [{label}]({rel})\n")
                f.write("\n")

    # Generate lang-map.json for zero-latency language switching.
    langMap = generateLangMap(NAV_PAGES_EN)
    with mkdocs_gen_files.open("lang-map.json", "w") as f:
        json.dump(langMap, f, ensure_ascii=False, separators=(",", ":"))

    # Mirror commonly referenced doc assets (images) so docs render cleanly.
    for src in REPO_ROOT.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in ASSET_EXTS:
            continue
        rel = src.relative_to(REPO_ROOT).as_posix()
        if shouldSkip(rel):
            continue
        with mkdocs_gen_files.open(rel, "wb") as f:
            f.write(src.read_bytes())


main()
