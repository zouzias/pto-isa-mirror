#!/usr/bin/env python3
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


FAIL_RE = re.compile(r"FAIL|ERROR|Traceback|FAILED|stream sync failed|retCode")
PASS_RE = re.compile(r"PASS rank=")
TRACE_RE = re.compile(
    r"frontMode=(?P<front_mode>\d+).*?"
    r"frontCase=(?P<case>\d+).*?"
    r"routeElems=(?P<route>\d+).*?"
    r"sortNeedCoreNum=(?P<sort_cores>\d+).*?"
    r"dispatchGatherMode=(?P<dispatch_gather>\d+)"
)
STAGE_HEADER_RE = re.compile(r"rank=(?P<rank>\d+) stageProfileEnvelope .* kernel_us=(?P<kernel>[0-9.]+)")
STAGE_LINE_RE = re.compile(
    r"\s+(?P<name>\w+) active=(?P<active>\d+) .* envelope_us=(?P<envelope>[0-9.]+) "
    r"max_core_us=(?P<max_core>[0-9.]+)"
)
FRONT_HEADER_RE = re.compile(r"rank=(?P<rank>\d+) frontStepDetailEnvelope")
FRONT_LINE_RE = re.compile(
    r"\s+(?P<name>\w+) active=(?P<active>\d+) .* envelope_us=(?P<envelope>[0-9.]+) "
    r"max_core_us=(?P<max_core>[0-9.]+)"
)
INACTIVE_LINE_RE = re.compile(r"\s+(?P<name>\w+) inactive")
CORE_HEADER_RE = re.compile(r"rank=(?P<rank>\d+) stageProfileCore")
AIV_CORE_HEADER_RE = re.compile(r"\s+aiv block=(?P<block>\d+) sub=(?P<sub>\d+)")
PROFILE_INTERVAL_RE = re.compile(r"(?P<name>\w+)=\([0-9.]+,[0-9.]+,(?P<duration>[0-9.]+)\)")


STAGE_NAMES = ["front", "dispatch", "gmm1", "swiglu", "gmm2", "combine", "unpermute"]
FRONT_STEP_NAMES = [
    "buildHistogram",
    "smallRouteMetadata",
    "smallWorkerRoute",
    "prepareBaseCursor",
    "scatterQuant",
    "exchangeCounts",
    "cumsum",
    "doneNotify",
    "sortVbs",
    "sortLocalMerge",
    "sortMiddleMerge",
    "sortOut",
    "metadata",
    "quant",
]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def nonzero(values: list[float]) -> list[float]:
    return [value for value in values if value > 0.0]


def summarize_rank_metric(rank_data: dict[int, dict], metric: str, stage: str | None = None) -> dict[str, float]:
    values: list[float] = []
    rank0 = 0.0
    for rank, data in sorted(rank_data.items()):
        if metric == "kernel":
            value = float(data.get("kernel_us", 0.0))
        elif metric == "stage":
            value = float(data.get("stages", {}).get(stage, {}).get("envelope_us", 0.0))
        elif metric == "front_step":
            value = float(data.get("front_steps", {}).get(stage, {}).get("envelope_us", 0.0))
        else:
            value = 0.0
        if rank == 0:
            rank0 = value
        values.append(value)
    active = nonzero(values)
    if not active:
        active = values
    return {
        "rank0": rank0,
        "max": max(active) if active else 0.0,
        "avg": statistics.fmean(active) if active else 0.0,
        "p95": percentile(active, 0.95),
    }


def parse_log(log_path: Path, label: str, m_value: int, run_idx: int, max_output_size: int) -> dict[str, str]:
    text = log_path.read_text(errors="replace")
    fail_markers = len(FAIL_RE.findall(text))
    pass_ranks = len(PASS_RE.findall(text))
    trace = TRACE_RE.search(text)

    rank_data: dict[int, dict] = {}
    core_data: dict[int, dict[str, list[float] | float]] = {}
    current_stage_rank: int | None = None
    current_front_rank: int | None = None
    current_core_rank: int | None = None
    inactive_counts = {name: 0 for name in FRONT_STEP_NAMES}

    for line in text.splitlines():
        stage_header = STAGE_HEADER_RE.search(line)
        if stage_header:
            current_stage_rank = int(stage_header.group("rank"))
            current_front_rank = None
            current_core_rank = None
            rank_data.setdefault(current_stage_rank, {"stages": {}, "front_steps": {}})
            rank_data[current_stage_rank]["kernel_us"] = float(stage_header.group("kernel"))
            continue

        front_header = FRONT_HEADER_RE.search(line)
        if front_header:
            current_front_rank = int(front_header.group("rank"))
            current_stage_rank = None
            current_core_rank = None
            rank_data.setdefault(current_front_rank, {"stages": {}, "front_steps": {}})
            continue

        core_header = CORE_HEADER_RE.search(line)
        if core_header:
            current_core_rank = int(core_header.group("rank"))
            current_stage_rank = None
            current_front_rank = None
            core_data.setdefault(current_core_rank, {})
            continue

        if current_stage_rank is not None:
            stage_line = STAGE_LINE_RE.match(line)
            if stage_line:
                rank_data[current_stage_rank]["stages"][stage_line.group("name")] = {
                    "active": int(stage_line.group("active")),
                    "envelope_us": float(stage_line.group("envelope")),
                    "max_core_us": float(stage_line.group("max_core")),
                }
            continue

        if current_front_rank is not None:
            front_line = FRONT_LINE_RE.match(line)
            if front_line:
                rank_data[current_front_rank]["front_steps"][front_line.group("name")] = {
                    "active": int(front_line.group("active")),
                    "envelope_us": float(front_line.group("envelope")),
                    "max_core_us": float(front_line.group("max_core")),
                }
                continue
            inactive_line = INACTIVE_LINE_RE.match(line)
            if inactive_line and inactive_line.group("name") in inactive_counts:
                inactive_counts[inactive_line.group("name")] += 1
            continue

        if current_core_rank is not None:
            core_line = AIV_CORE_HEADER_RE.match(line)
            if core_line:
                entry = core_data.setdefault(current_core_rank, {})
                is_core0 = int(core_line.group("block")) == 0 and int(core_line.group("sub")) == 0
                for interval in PROFILE_INTERVAL_RE.finditer(line):
                    name = interval.group("name")
                    duration = float(interval.group("duration"))
                    keys: list[str] = []
                    if name in {"front", "dispatch"}:
                        keys.append(name)
                    if name == "scatterQuant":
                        keys.append("scatter")
                    if name in FRONT_STEP_NAMES:
                        keys.append(f"step_{name}")
                    for key in keys:
                        durations = entry.setdefault(key, [])
                        if isinstance(durations, list):
                            durations.append(duration)
                        if is_core0:
                            entry[f"{key}_core0"] = duration

    if fail_markers != 0 or pass_ranks != 8 or not rank_data:
        raise RuntimeError(
            f"{log_path}: invalid run fail_markers={fail_markers} pass_ranks={pass_ranks} "
            f"profile_ranks={len(rank_data)}"
        )

    row: dict[str, str] = {
        "label": label,
        "m": str(m_value),
        "run": str(run_idx),
        "max_output_size": str(max_output_size),
        "log_path": str(log_path),
        "pass_ranks": str(pass_ranks),
        "fail_markers": str(fail_markers),
        "front_mode": trace.group("front_mode") if trace else "",
        "dispatch_gather_mode": trace.group("dispatch_gather") if trace else "",
        "front_case": trace.group("case") if trace else "",
        "route_elems": trace.group("route") if trace else "",
        "sort_need_core_num": trace.group("sort_cores") if trace else "",
    }

    for prefix, metric, name in [("kernel", "kernel", None)]:
        stats = summarize_rank_metric(rank_data, metric, name)
        row[f"{prefix}_rank0_us"] = fmt(stats["rank0"])
        row[f"{prefix}_max_us"] = fmt(stats["max"])
        row[f"{prefix}_avg_us"] = fmt(stats["avg"])
        row[f"{prefix}_p95_us"] = fmt(stats["p95"])

    for stage in STAGE_NAMES:
        stats = summarize_rank_metric(rank_data, "stage", stage)
        row[f"{stage}_rank0_us"] = fmt(stats["rank0"])
        row[f"{stage}_max_us"] = fmt(stats["max"])
        row[f"{stage}_avg_us"] = fmt(stats["avg"])
        row[f"{stage}_p95_us"] = fmt(stats["p95"])
        active = rank_data.get(0, {}).get("stages", {}).get(stage, {}).get("active", "")
        row[f"{stage}_rank0_active"] = fmt(active)

    for step in FRONT_STEP_NAMES:
        stats = summarize_rank_metric(rank_data, "front_step", step)
        row[f"{step}_rank0_us"] = fmt(stats["rank0"])
        row[f"{step}_max_us"] = fmt(stats["max"])
        row[f"{step}_avg_us"] = fmt(stats["avg"])
        row[f"{step}_p95_us"] = fmt(stats["p95"])
        active = rank_data.get(0, {}).get("front_steps", {}).get(step, {}).get("active", "")
        row[f"{step}_rank0_active"] = fmt(active)
        row[f"{step}_inactive_ranks"] = str(inactive_counts[step])

    rank0_core = core_data.get(0, {})
    core_metric_names = [("front", "front"), ("dispatch", "dispatch"), ("scatter", "scatter")]
    core_metric_names.extend((step, f"step_{step}") for step in FRONT_STEP_NAMES)
    for name, key in core_metric_names:
        durations = rank0_core.get(key, [])
        if isinstance(durations, list) and durations:
            active_durations = nonzero(durations)
            median = statistics.median(active_durations) if active_durations else 0.0
            core0 = float(rank0_core.get(f"{key}_core0", 0.0))
            row[f"{name}_core0_us"] = fmt(core0)
            row[f"{name}_core_median_us"] = fmt(median)
            row[f"{name}_core0_to_median"] = fmt(core0 / median if median > 0.0 else 0.0)
        else:
            row[f"{name}_core0_us"] = ""
            row[f"{name}_core_median_us"] = ""
            row[f"{name}_core0_to_median"] = ""

    front_rank0 = float(row.get("front_rank0_us") or 0.0)
    for step in FRONT_STEP_NAMES:
        step_rank0 = float(row.get(f"{step}_rank0_us") or 0.0)
        row[f"{step}_front_ratio_rank0"] = fmt(step_rank0 / front_rank0 if front_rank0 > 0.0 else 0.0)

    return row


def fieldnames() -> list[str]:
    names = [
        "label",
        "m",
        "run",
        "max_output_size",
        "log_path",
        "pass_ranks",
        "fail_markers",
        "front_mode",
        "dispatch_gather_mode",
        "front_case",
        "route_elems",
        "sort_need_core_num",
        "kernel_rank0_us",
        "kernel_max_us",
        "kernel_avg_us",
        "kernel_p95_us",
    ]
    for stage in STAGE_NAMES:
        names.extend(
            [f"{stage}_rank0_us", f"{stage}_max_us", f"{stage}_avg_us", f"{stage}_p95_us", f"{stage}_rank0_active"]
        )
    for step in FRONT_STEP_NAMES:
        names.extend(
            [
                f"{step}_rank0_us",
                f"{step}_max_us",
                f"{step}_avg_us",
                f"{step}_p95_us",
                f"{step}_rank0_active",
                f"{step}_inactive_ranks",
            ]
        )
    names.extend(
        [
            "front_core0_us",
            "front_core_median_us",
            "front_core0_to_median",
            "dispatch_core0_us",
            "dispatch_core_median_us",
            "dispatch_core0_to_median",
            "scatter_core0_us",
            "scatter_core_median_us",
            "scatter_core0_to_median",
        ]
    )
    for step in FRONT_STEP_NAMES:
        names.extend(
            [f"{step}_core0_us", f"{step}_core_median_us", f"{step}_core0_to_median", f"{step}_front_ratio_rank0"]
        )
    return names


def ensure_tsv(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.exists():
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames(), delimiter="\t")
            writer.writeheader()


def append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames(), delimiter="\t")
        for row in rows:
            writer.writerow(row)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def median_float(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row.get(key, "")) for row in rows if row.get(key)]
    return statistics.median(values) if values else 0.0


def write_summary(out_dir: Path) -> None:
    stage_a = read_tsv(out_dir / "v5_stage_a.tsv")
    stage_b = read_tsv(out_dir / "v5_stage_b.tsv")
    labels = [("stage_a", stage_a), ("stage_b", stage_b)]
    lines = ["# V8 front dispatch gather performance summary", ""]
    for label, rows in labels:
        lines.append(f"## {label}")
        if not rows:
            lines.append("")
            lines.append("No runs recorded.")
            lines.append("")
            continue
        lines.append("")
        lines.append(
            "| m | runs | kernel_med_us | front_med_us | dispatch_med_us | scatter_med_us | exchange_med_us | "
            "sort_vbs_med_us | local_merge_med_us | middle_merge_med_us | sort_out_med_us | metadata_med_us | "
            "quant_med_us |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for m_value in sorted({int(row["m"]) for row in rows}):
            subset = [row for row in rows if int(row["m"]) == m_value]
            lines.append(
                f"| {m_value} | {len(subset)} | {median_float(subset, 'kernel_max_us'):.2f} | "
                f"{median_float(subset, 'front_max_us'):.2f} | {median_float(subset, 'dispatch_max_us'):.2f} | "
                f"{median_float(subset, 'scatterQuant_max_us'):.2f} | "
                f"{median_float(subset, 'exchangeCounts_max_us'):.2f} | "
                f"{median_float(subset, 'sortVbs_max_us'):.2f} | "
                f"{median_float(subset, 'sortLocalMerge_max_us'):.2f} | "
                f"{median_float(subset, 'sortMiddleMerge_max_us'):.2f} | "
                f"{median_float(subset, 'sortOut_max_us'):.2f} | "
                f"{median_float(subset, 'metadata_max_us'):.2f} | "
                f"{median_float(subset, 'quant_max_us'):.2f} |"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- stage_a is the baseline collected before B1-B7 performance edits.")
    lines.append("- stage_b is populated by rerunning this harness after each performance edit.")
    lines.append(
        "- Stage B logs expose VBS/local merge/middle merge/SortOut/metadata/quant as front-step profile rows."
    )
    lines.append(
        "- Older stage_a TSV files may not contain the split columns; missing values are reported as 0 in this"
    )
    lines.append("  summary and should be compared against scatterQuant for the pre-instrumentation baseline.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def run_case(args: argparse.Namespace, m_value: int, run_idx: int, max_output_size: int, log_path: Path) -> None:
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = args.ascend_home
    env["ASCEND_PROCESS_LOG_PATH"] = str(args.out_dir / f"ascend_plog_{args.label}_m{m_value}_run{run_idx}")
    env["DISPATCH_FFN_COMBINE_V8_TRACE"] = "1"
    env["DISPATCH_FFN_COMBINE_V8_STAGE_PROFILE"] = "1"
    env["DISPATCH_FFN_COMBINE_V8_WARMUP_ITERS"] = str(args.warmup_iters)
    env["DISPATCH_FFN_COMBINE_V8_MEASURE_ITERS"] = str(args.measure_iters)
    cmd = [
        "bash",
        str(args.run_sh),
        "--world-size",
        str(args.world_size),
        "--m",
        str(m_value),
        "--k",
        str(args.k),
        "--n",
        str(args.n),
        "--topk",
        str(args.topk),
        "--experts",
        str(args.experts),
        "--max-output-size",
        str(max_output_size),
        "--reuse-data",
    ]
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=args.project_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"{log_path}: command failed with exit code {proc.returncode}")


def parse_m_list(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item]
    if not values:
        raise ValueError("m-list must not be empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("out/front_dispatch_gather_perf"))
    parser.add_argument("--label", choices=["stage_a", "stage_b"], default="stage_a")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--m-list", default="512,1024,2048")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=7168)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--ascend-home", default="/home/ntlab/liulei/can/cann")
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--measure-iters", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()

    args.project_dir = args.project_dir.resolve()
    args.out_dir = (args.project_dir / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    args.run_sh = args.project_dir / "run.sh"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stage_a_tsv = args.out_dir / "v5_stage_a.tsv"
    stage_b_tsv = args.out_dir / "v5_stage_b.tsv"
    ensure_tsv(stage_a_tsv, args.overwrite and args.label == "stage_a")
    ensure_tsv(stage_b_tsv, args.overwrite and args.label == "stage_b")

    target_tsv = stage_a_tsv if args.label == "stage_a" else stage_b_tsv
    rows: list[dict[str, str]] = []
    for m_value in parse_m_list(args.m_list):
        max_output_size = 8194 if m_value <= 128 else 81940
        for run_idx in range(args.runs):
            log_path = args.out_dir / f"{args.label}_m{m_value}_run{run_idx}.log"
            if not args.parse_only:
                print(f"[front_dispatch_gather_perf] run label={args.label} m={m_value} run={run_idx}", flush=True)
                run_case(args, m_value, run_idx, max_output_size, log_path)
            row = parse_log(log_path, args.label, m_value, run_idx, max_output_size)
            rows.append(row)
            print(
                "[front_dispatch_gather_perf] parsed "
                f"m={m_value} run={run_idx} kernel_max_us={row['kernel_max_us']} "
                f"front_max_us={row['front_max_us']} dispatch_max_us={row['dispatch_max_us']}",
                flush=True,
            )

    append_rows(target_tsv, rows)
    write_summary(args.out_dir)
    print(f"[front_dispatch_gather_perf] wrote {target_tsv}")
    print(f"[front_dispatch_gather_perf] wrote {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[front_dispatch_gather_perf] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
