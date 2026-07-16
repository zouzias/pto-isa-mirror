#!/usr/bin/env python3
"""Generate V8 2048 pipeline SVGs from gmm1_expert_profile_2048.log."""

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_CANDIDATES = [
    BASE_DIR / "reprofile_combine_drain.stdout",
    BASE_DIR / "gmm1_expert_profile_2048.log",
    BASE_DIR / "out" / "reprofile_combine_drain.stdout",
    BASE_DIR / "out" / "gmm1_expert_profile_2048.log",
]
LOG = next((path for path in LOG_CANDIDATES if path.exists()), LOG_CANDIDATES[0])
OUT_MACRO = BASE_DIR / "v5_2048_overlap_pipeline.svg"
OUT_GMM1 = BASE_DIR / "v5_2048_gmm1_expert_pipeline.svg"

T0, T1 = 70.0, 970.0  # detailed chart
T0M, T1M = 70.0, 940.0  # macro chart

COLORS = [
    "#5B8DEF",
    "#4ECDC4",
    "#FF6B6B",
    "#FFD166",
    "#06D6A0",
    "#118AB2",
    "#9B5DE5",
    "#E07A5F",
    "#3D5A80",
    "#81B29A",
    "#F2CC8F",
    "#E76F51",
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
]


def tx(t: float, t_max: float, x0: float, x1: float) -> float:
    return x0 + (t / t_max) * (x1 - x0)


def tx_gmm1(t: float, t_max: float = 2400.0) -> float:
    return tx(t, t_max, T0, T1)


def tx_macro(t: float, kernel_us: float) -> float:
    return tx(t, kernel_us, T0M, T1M)


def parse_expert_wall(rank0: str, header: str) -> list[tuple[int, float, float, float]]:
    """Parse lines under `rank=0 <header> ...` section."""
    m = re.search(rf"rank=0 {re.escape(header)}[^\n]*\n((?:  expert=\d+[^\n]*\n)+)", rank0)
    if not m:
        return []
    out: list[tuple[int, float, float, float]] = []
    for line in m.group(1).splitlines():
        em = re.search(r"expert=(\d+) start_us=([\d.]+) end_us=([\d.]+)(?: wall_us=([\d.]+))?", line)
        if not em:
            continue
        ex = int(em.group(1))
        s, e = float(em.group(2)), float(em.group(3))
        wall = float(em.group(4)) if em.group(4) else e - s
        out.append((ex, s, e, wall))
    out.sort(key=lambda x: x[0])
    return out


def parse_segment_wall(rank0: str, header: str = "swigluSegmentWall") -> list[tuple[int, float, float, float]]:
    m = re.search(rf"rank=0 {re.escape(header)}[^\n]*\n((?:  segment=\d+[^\n]*\n)+)", rank0)
    if not m:
        return []
    out: list[tuple[int, float, float, float]] = []
    for line in m.group(1).splitlines():
        sm = re.search(r"segment=(\d+) start_us=([\d.]+) end_us=([\d.]+)(?: wall_us=([\d.]+))?", line)
        if not sm:
            continue
        seg = int(sm.group(1))
        s, e = float(sm.group(2)), float(sm.group(3))
        wall = float(sm.group(4)) if sm.group(4) else e - s
        out.append((seg, s, e, wall))
    out.sort(key=lambda x: x[0])
    return out


def parse_stage_ready_end(rank0: str, stage: str) -> tuple[float, float, float]:
    """Return (stage_start, ready_start, end) from aiv block=0 sub=0 stageProfileCore."""
    aiv_line = re.search(r"aiv block=0 sub=0 kernel=\([^)]+\) (.+)$", rank0, re.M).group(1)
    m = re.search(rf"{stage}=\(([^)]+)\)", aiv_line)
    p = m.group(1).split(",")
    stage_start = float(p[0])
    ready = float(p[1]) if p[1] != "-" else stage_start
    end = float(p[2])
    return stage_start, ready, end


def parse_detail_end_us(rank0: str, detail_name: str) -> float | None:
    m = re.search(rf"^\s*{re.escape(detail_name)} active=\d+ .* end_us=([\d.]+)", rank0, re.M)
    return float(m.group(1)) if m else None


def adjust_final_combine_expert(
    items: list[tuple[int, float, float, float]], large_end: float | None
) -> list[tuple[int, float, float, float]]:
    if not items or large_end is None:
        return items
    out = list(items)
    idx, start, end, _dur = out[-1]
    if large_end <= end:
        return out
    prev_durations = [e - s for _idx, s, e, _dur in out[:-1]]
    typical = sorted(prev_durations)[len(prev_durations) // 2] if prev_durations else 0.0
    if typical == 0.0 or end - start < typical * 0.5:
        out[-1] = (idx, start, large_end, large_end - start)
    return out


def build_combine_chart(
    combine_aiv0: list[tuple[int, float, float, float]],
    combine_ready: float,
    combine_rank: list[tuple[int, float, float, float]] | None,
) -> list[tuple[int, float, float, float]]:
    """Combine Cn on aiv0 sub=0: E0 starts at combine readyStart; bars must not overlap."""
    if not combine_aiv0:
        return []
    rank_by_ex = {ex: (cs, ce) for ex, cs, ce, _ in combine_rank} if combine_rank else {}

    out: list[tuple[int, float, float, float]] = []
    for ex, cs, ce, _ in combine_aiv0:
        if ex in rank_by_ex:
            start, end = rank_by_ex[ex]
        elif ex == 0:
            start = combine_ready
            end = ce
        else:
            start, end = cs, ce
        if end <= start:
            end = start + 1.0
        out.append((ex, start, end, end - start))
    return out


def gmm2_ready_starts(gmm2_experts: list[tuple[int, float, float, float]]) -> dict[int, float]:
    return {ex: s for ex, s, _e, _w in gmm2_experts}


def split_equal_experts(start: float, end: float, count: int = 16) -> list[tuple[int, float, float, float]]:
    if count <= 0 or end <= start:
        return []
    seg = (end - start) / float(count)
    return [(i, start + i * seg, start + (i + 1) * seg, seg) for i in range(count)]


def parse_log(text: str) -> dict:
    rank0 = text.split("rank=1 ")[0]

    km = re.search(r"rank=0 stageProfileEnvelope[^\n]* kernel_us=([\d.]+)", rank0)
    kernel_us = float(km.group(1)) if km else 4404.0

    gmm1 = re.search(r"aic block=0.*gmm1=\(([^)]+)\)", rank0).group(1).split(",")
    gmm1_ready = float(gmm1[1])
    gmm1_end = float(gmm1[2])

    gmm2 = re.search(r"aic block=0.*gmm2=\(([^)]+)\)", rank0).group(1).split(",")
    gmm2_start, gmm2_end = map(float, gmm2[:3:2])  # start, end (skip ready)

    aiv_line = re.search(r"aiv block=0 sub=0 kernel=\([^)]+\) (.+)$", rank0, re.M).group(1)

    def parse_stage(name: str) -> tuple[float, float]:
        m = re.search(rf"{name}=\(([^)]+)\)", aiv_line)
        p = m.group(1).split(",")
        a, b, c = p[0], p[1], p[2]
        if b == "-":
            return float(a), float(c)
        return float(b), float(c)

    front_s, front_e = parse_stage("front")
    dispatch_s, dispatch_e = parse_stage("dispatch")
    swiglu_s, swiglu_e = parse_stage("swiglu")
    combine_s, combine_e = parse_stage("combine")
    unpermute_s, unpermute_e = parse_stage("unpermute")
    _combine_stage_start, combine_ready, _combine_stage_end = parse_stage_ready_end(rank0, "combine")

    # Prefer direct wall-clock sections (aic0 / aiv0)
    gmm1_wall = parse_expert_wall(rank0, "gmm1ExpertWall")
    gmm2_wall = parse_expert_wall(rank0, "gmm2ExpertWall")
    dispatch_wall = parse_expert_wall(rank0, "dispatchExpertWall")
    combine_wall_aiv0 = parse_expert_wall(rank0, "combineExpertWall")
    combine_wall_rank = parse_expert_wall(rank0, "combineExpertWallRank")
    combine_wall_aligned = parse_expert_wall(rank0, "combineExpertWallAligned")
    combine_wall = combine_wall_aligned or combine_wall_rank or combine_wall_aiv0
    swiglu_wall_aiv0 = parse_segment_wall(rank0, "swigluSegmentWall")
    swiglu_wall = parse_segment_wall(rank0, "swigluSegmentWallRank") or swiglu_wall_aiv0
    combine_large_end = parse_detail_end_us(rank0, "largeGroup") or parse_detail_end_us(rank0, "directLargeTotal")

    if not gmm1_wall:
        gmm1_wall = split_equal_experts(gmm1_ready, gmm1_end, 16)

    # Fallback dispatch: equal segments across stage window
    if not dispatch_wall:
        seg = (dispatch_e - dispatch_s) / 16.0
        dispatch_wall = [(i, dispatch_s + i * seg, dispatch_s + (i + 1) * seg, seg) for i in range(16)]

    # Fallback swiglu: split stage window in half
    if not swiglu_wall:
        mid = (swiglu_s + swiglu_e) / 2.0
        swiglu_wall = [(0, swiglu_s, mid, mid - swiglu_s), (1, mid, swiglu_e, swiglu_e - mid)]

    # Fallback gmm2: single expert spanning whole stage (shown as one bar)
    if not gmm2_wall:
        gmm2_wall = split_equal_experts(gmm2_start, gmm2_end, 16)

    # Fallback combine: equal segments
    if not combine_wall:
        seg = (combine_e - combine_s) / 16.0
        combine_wall = [(i, combine_s + i * seg, combine_s + (i + 1) * seg, seg) for i in range(16)]

    gmm1_experts = [(ex, s, e, w) for ex, s, e, w in gmm1_wall]

    combine_experts_chart = build_combine_chart(combine_wall_aiv0, combine_ready, combine_wall_rank or None)
    if not combine_experts_chart:
        combine_experts_chart = combine_wall_aligned or combine_wall_rank or combine_wall_aiv0 or combine_wall
    combine_experts_chart = adjust_final_combine_expert(combine_experts_chart, combine_large_end)
    gmm2_ready = gmm2_ready_starts(gmm2_wall)
    unpermute_chart_s = unpermute_s
    if combine_experts_chart:
        # Attribute the post-combine full-sync gap to Unpermute in the visualization.
        unpermute_chart_s = min(unpermute_s, combine_experts_chart[-1][2])

    return {
        "kernel_us": kernel_us,
        "gmm1_ready": gmm1_ready,
        "gmm1_end": gmm1_end,
        "gmm2_start": gmm2_start,
        "gmm2_end": gmm2_end,
        "front": (front_s, front_e),
        "dispatch": (dispatch_s, dispatch_e),
        "swiglu": (swiglu_s, swiglu_e),
        "combine": (combine_s, combine_e),
        "unpermute": (unpermute_s, unpermute_e),
        "unpermute_chart": (unpermute_chart_s, unpermute_e),
        "combine_ready": combine_ready,
        "gmm1_experts": gmm1_experts,
        "gmm2_experts": gmm2_wall,
        "dispatch_experts": dispatch_wall,
        "swiglu_segments": swiglu_wall,
        "swiglu_segments_aiv0": swiglu_wall_aiv0,
        "combine_experts": combine_experts_chart,
        "combine_experts_raw": combine_wall,
        "combine_experts_aiv0": combine_wall_aiv0,
        "gmm2_ready_starts": gmm2_ready,
        "has_wall_logs": bool(parse_expert_wall(rank0, "gmm2ExpertWall")),
        "has_rank_envelope": bool(parse_expert_wall(rank0, "combineExpertWallRank")),
    }


def axis_macro(kernel_us: float) -> str:
    ticks = [0, 300, 800, 1300, 2000, 2300, 3200, 4000, int(kernel_us)]
    lines = [
        f'<line x1="{T0M}" y1="55" x2="{T1M}" y2="55" stroke="#ccc"/>',
        f'<text x="{(T0M + T1M) / 2}" y="75" text-anchor="middle" fill="#666" font-size="10">time (us)</text>',
    ]
    for t in ticks:
        x = tx_macro(t, kernel_us)
        lines.append(f'<line x1="{x:.1f}" y1="55" x2="{x:.1f}" y2="60" stroke="#ccc"/>')
        lines.append(f'<text x="{x:.1f}" y="50" text-anchor="middle" fill="#888" font-size="9">{t}</text>')
    return "\n".join(lines)


def axis_gmm1(t_max: float) -> str:
    ticks = list(range(0, int(t_max) + 1, 300))
    lines = [
        f'<line x1="{T0}" y1="62" x2="{T1}" y2="62" stroke="#aaa"/>',
        f'<text x="{(T0 + T1) / 2}" y="80" text-anchor="middle" fill="#666" font-size="10">time (us)</text>',
    ]
    for t in ticks:
        x = tx_gmm1(t, t_max)
        lines.append(f'<line x1="{x:.1f}" y1="62" x2="{x:.1f}" y2="66" stroke="#aaa"/>')
        lines.append(f'<text x="{x:.1f}" y="56" text-anchor="middle" fill="#888" font-size="9">{t}</text>')
    return "\n".join(lines)


def draw_expert_row(
    parts: list[str],
    items: list[tuple[int, float, float, float]],
    y: int,
    h: int,
    tx_fn,
    row_label: str,
    id_prefix: str = "E",
    labels_above: bool = False,
    narrow_label_at_start: bool = False,
) -> None:
    parts.append(f'<text x="14" y="{y + h // 2 + 4}" fill="#aaa" font-size="9">{row_label}</text>')
    labels: list[tuple[str, float, float]] = []
    for idx, s, e, _dur in items:
        x0 = tx_fn(s)
        x1 = tx_fn(e)
        w = max(x1 - x0, 0.8)
        c = COLORS[idx % len(COLORS)]
        parts.append(
            f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" fill="{c}" stroke="#181818" stroke-width="0.6"/>'
        )
        labels.append((f"{id_prefix}{idx}", x0, (x0 + x1) / 2.0, w))
    for label, x0, xc, w in labels:
        if narrow_label_at_start and w < 14.0:
            parts.append(
                f'<text x="{x0:.1f}" y="{y - 3}" text-anchor="start" '
                f'fill="#eee" font-size="7" font-weight="bold">{label}</text>'
            )
        elif labels_above or w < 11.0:
            parts.append(
                f'<text x="{xc:.1f}" y="{y - 3}" text-anchor="middle" '
                f'fill="#ddd" font-size="7" font-weight="bold">{label}</text>'
            )
        else:
            parts.append(
                f'<text x="{xc:.1f}" y="{y + h / 2 + 2}" text-anchor="middle" '
                f'fill="#fff" font-size="7" font-weight="bold">{label}</text>'
            )


def draw_expert_labels(parts: list[str], items: list[tuple[int, float, float, float]], y: int, prefix: str) -> int:
    x_lbl = 96.0
    line_h = 14
    for idx, s, e, _ in items:
        txt = f"{prefix}{idx}:{s:.0f}-{e:.0f}"
        tw = len(txt) * 4.8
        if x_lbl + tw > 950:
            y += line_h
            x_lbl = 96.0
        parts.append(f'<text x="{x_lbl:.1f}" y="{y}" fill="#888" font-size="7">{txt}</text>')
        x_lbl += tw + 12
    return y + line_h


def gen_macro(d: dict) -> str:
    ku = d["kernel_us"]
    txm = lambda t: tx_macro(t, ku)

    def bar(x0, x1, y, h, fill, label, fs=8, rx=4):
        w = max(x1 - x0, 1.0)
        return (
            f'<rect x="{x0:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="{rx}" fill="{fill}"/>'
            f'<text x="{(x0 + x1) / 2:.1f}" y="{y + h / 2 + 3}" text-anchor="middle" fill="#fff" font-size="{fs}">{label}</text>'
        )

    h = 600
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="{h}" viewBox="0 0 980 {h}">',
        '<rect width="100%" height="100%" fill="#181818"/>',
        '<text x="490" y="24" text-anchor="middle" fill="#E4E4E4" font-size="14" font-weight="bold">'
        "2048 token pipeline · AIV / AIC per-expert wall</text>",
        axis_macro(ku),
        '<text x="14" y="108" fill="#ccc" font-weight="bold" font-size="11">AIV</text>',
    ]

    fs, fe = d["front"]
    parts.append(bar(txm(fs), txm(fe), 90, 24, "#5B8DEF", f"Front {fs:.0f}-{fe:.0f}"))

    draw_expert_row(parts, d["dispatch_experts"], 122, 24, txm, "Dispatch", "D")
    draw_expert_row(parts, d["swiglu_segments"], 154, 24, txm, "SwiGLU", "S", labels_above=True)
    draw_expert_row(parts, d["combine_experts"], 186, 24, txm, "Combine", "C", narrow_label_at_start=True)
    us, ue = d["unpermute_chart"]
    parts.append(bar(txm(us), txm(ue), 218, 24, "#7C6A0A", f"Unpermute {us:.0f}-{ue:.0f}"))

    parts.append('<text x="14" y="264" fill="#ccc" font-weight="bold" font-size="11">AIC</text>')
    draw_expert_row(parts, d["gmm1_experts"], 272, 28, txm, "GMM1", "E")
    draw_expert_row(parts, d["gmm2_experts"], 310, 28, txm, "GMM2", "E")

    lbl_y = 356
    for label, items, prefix in [
        ("Dispatch:", d["dispatch_experts"], "D"),
        ("SwiGLU:", d["swiglu_segments"], "S"),
        ("Combine:", d["combine_experts"], "C"),
        ("GMM1:", d["gmm1_experts"], "E"),
        ("GMM2:", d["gmm2_experts"], "E"),
    ]:
        parts.append(f'<text x="14" y="{lbl_y}" fill="#888" font-size="8">{label}</text>')
        lbl_y = draw_expert_labels(parts, items, lbl_y, prefix) + 6

    ge = d["gmm1_end"]
    g2e = d["gmm2_end"]
    ds, de = d["dispatch"]
    ss, se = d["swiglu"]
    cs, ce = d["combine"]
    g2s = d["gmm2_start"]
    gr = d["gmm1_ready"]

    ov_y = lbl_y + 8
    parts.append(f'<text x="14" y="{ov_y}" fill="#aaa" font-size="10" font-weight="bold">Overlap:</text>')
    overlaps = [
        ("GMM1∥Dispatch", max(gr, ds), min(ge, de)),
        ("GMM1∥SwiGLU", max(gr, ss), min(ge, se)),
        ("GMM2∥SwiGLU", max(g2s, ss), min(g2e, se)),
        ("GMM2∥Combine", max(g2s, cs), min(g2e, ce)),
    ]
    for j, (name, a, b) in enumerate(overlaps):
        if b > a:
            parts.append(
                f'<text x="14" y="{ov_y + 16 + j * 14}" fill="#888" font-size="9">'
                f"{name}: {a:.0f}-{b:.0f} ({b - a:.0f}us)</text>"
            )

    parts.append("</svg>")
    return "\n".join(parts)


def gen_gmm1(d: dict) -> str:
    ku = max(d["gmm1_end"], d["gmm2_end"], d["combine"][1]) + 200
    txg = lambda t: tx_gmm1(t, ku)
    fs, fe = d["front"]
    us, ue = d["unpermute_chart"]

    h = 1080
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{h}" viewBox="0 0 1000 {h}">',
        '<rect width="100%" height="100%" fill="#181818"/>',
        '<text x="500" y="28" text-anchor="middle" fill="#E4E4E4" font-size="15" font-weight="bold">'
        "2048 · full pipeline expert/segment wall</text>",
        axis_gmm1(ku),
        '<text x="12" y="113" fill="#ccc" font-size="11" font-weight="bold">AIV</text>',
        f'<rect x="{txg(fs):.1f}" y="95" width="{txg(fe) - txg(fs):.1f}" height="22" rx="3" fill="#5B8DEF"/>',
        f'<text x="{txg((fs + fe) / 2):.1f}" y="110" text-anchor="middle" fill="#fff" font-size="8">Front</text>',
        f'<rect x="{txg(us):.1f}" y="125" width="{txg(ue) - txg(us):.1f}" height="22" rx="3" fill="#7C6A0A"/>',
        f'<text x="{txg((us + ue) / 2):.1f}" y="140" text-anchor="middle" fill="#fff" font-size="8">Unpermute</text>',
    ]

    y = 155
    for title, items, prefix in [
        ("Dispatch", d["dispatch_experts"], "D"),
        ("SwiGLU", d["swiglu_segments"], "S"),
        ("Combine", d["combine_experts"], "C"),
    ]:
        parts.append(f'<text x="12" y="{y + 14}" fill="#ccc" font-size="10">{title}</text>')
        draw_expert_row(
            parts,
            items,
            y,
            18,
            txg,
            "",
            prefix,
            labels_above=(title == "SwiGLU"),
            narrow_label_at_start=(title == "Combine"),
        )
        y += 28

    y += 8
    parts.append(f'<text x="12" y="{y + 14}" fill="#ccc" font-size="11" font-weight="bold">AIC GMM1</text>')
    y += 22
    for i, (ex, s, e, dur) in enumerate(d["gmm1_experts"]):
        yy = y + i * 22
        w = max(txg(e) - txg(s), 2.0)
        x = txg(s)
        c = COLORS[i % len(COLORS)]
        parts.append(f'<text x="14" y="{yy + 11}" fill="#aaa" font-size="9">G1_{ex}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="16" rx="2" fill="{c}"/>')
        parts.append(
            f'<text x="{x + 4:.1f}" y="{yy + 11}" fill="#fff" font-size="8">{s:.0f}-{e:.0f} ({dur:.0f}us)</text>'
        )

    y_g2 = y + len(d["gmm1_experts"]) * 22 + 16
    parts.append(f'<text x="12" y="{y_g2 + 14}" fill="#ccc" font-size="11" font-weight="bold">AIC GMM2</text>')
    y_g2 += 22
    for i, (ex, s, e, dur) in enumerate(d["gmm2_experts"]):
        yy = y_g2 + i * 22
        w = max(txg(e) - txg(s), 2.0)
        x = txg(s)
        c = COLORS[i % len(COLORS)]
        parts.append(f'<text x="14" y="{yy + 11}" fill="#aaa" font-size="9">G2_{ex}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="16" rx="2" fill="{c}"/>')
        parts.append(
            f'<text x="{x + 4:.1f}" y="{yy + 11}" fill="#fff" font-size="8">{s:.0f}-{e:.0f} ({dur:.0f}us)</text>'
        )

    parts.append(
        f'<text x="70" y="{h - 20}" fill="#888" font-size="9">'
        "Dispatch/SwiGLU/Combine and GMM1/GMM2 from expertWall sections</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    text = LOG.read_text()
    data = parse_log(text)
    OUT_MACRO.write_text(gen_macro(data))
    OUT_GMM1.write_text(gen_gmm1(data))
    print("Wrote", OUT_MACRO.name, OUT_GMM1.name, f"kernel_us={data['kernel_us']:.2f}")
    for label, items in [
        ("GMM1", data["gmm1_experts"]),
        ("GMM2", data["gmm2_experts"]),
        ("Dispatch", data["dispatch_experts"]),
        ("Combine", data["combine_experts"]),
    ]:
        if items:
            ex0 = items[0]
            print(f"  {label}[0]: {ex0[1]:.2f}-{ex0[2]:.2f} ({ex0[3]:.2f}us)")


if __name__ == "__main__":
    main()
