#!/usr/bin/env python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Performance benchmark visualization: PTO AllGather-GEMM vs SHMEM AllGather-MatMul.

Generates three charts per matrix shape:
  1. Time comparison (Sequential breakdown vs PTO Fused vs SHMEM Fused)
  2. Overlap percentage comparison
  3. Bandwidth comparison

Usage:
  python3 scripts/plot_benchmark.py benchmark_results/benchmark_data.csv
  python3 scripts/plot_benchmark.py benchmark_results/benchmark_data.csv --output-dir ./charts
  python3 scripts/plot_benchmark.py  # uses sample data for demo
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def parse_csv(csv_path):
    """Parse benchmark CSV into structured data grouped by shape label."""
    data = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["shape_label"]
            entry = {
                "M": int(row["M"]),
                "K": int(row["K"]),
                "N": int(row["N"]),
                "pe_size": int(row["pe_size"]),
                "operator": row["operator"],
                "mode": row["mode"],
                "time_us": float(row["time_us"]),
                "tflops": float(row["tflops"]),
                "comm_bw_gbps": float(row["comm_bw_gbps"]),
                "comm_time_us": float(row["comm_time_us"]),
                "compute_time_us": float(row["compute_time_us"]),
                "overlap_pct": float(row["overlap_pct"]),
            }
            data[label].append(entry)
    return data


def generate_sample_data():
    """Generate realistic sample data for demonstration."""
    shapes = [
        ("1024x1024x512", 1024, 1024, 512),
        ("1024x2048x1024", 1024, 2048, 1024),
        ("2048x2048x1024", 2048, 2048, 1024),
        ("4096x4096x2048", 4096, 4096, 2048),
        ("8192x4096x4096", 8192, 4096, 4096),
        ("16384x4096x4096", 16384, 4096, 4096),
    ]

    pe_size = 2
    peak_tflops = 320.0
    comm_bw_est = 50.0

    data = defaultdict(list)
    for label, m, k, n in shapes:
        gemm_flops = 2.0 * m * k * n
        comm_bytes = (m / pe_size) * k * 2 * (pe_size - 1)

        pure_compute_us = gemm_flops / (peak_tflops * 1e12) * 1e6
        pure_comm_us = comm_bytes / (comm_bw_est * 1e9) * 1e6
        seq_total_us = pure_compute_us + pure_comm_us

        pto_fused_us = max(pure_compute_us, pure_comm_us) * 1.08
        shmem_fused_us = max(pure_compute_us, pure_comm_us) * 1.22

        pto_overlap = max(0, (1 - pto_fused_us / seq_total_us) * 100)
        shmem_overlap = max(0, (1 - shmem_fused_us / seq_total_us) * 100)

        pto_tflops = gemm_flops / (pto_fused_us * 1e-6) / 1e12
        shmem_tflops = gemm_flops / (shmem_fused_us * 1e-6) / 1e12
        pto_bw = comm_bytes / (pto_fused_us * 1e-6) / 1e9
        shmem_bw = comm_bytes / (shmem_fused_us * 1e-6) / 1e9

        data[label].append({
            "M": m, "K": k, "N": n, "pe_size": pe_size,
            "operator": "PTO", "mode": "sequential",
            "time_us": seq_total_us, "tflops": 0, "comm_bw_gbps": 0,
            "comm_time_us": pure_comm_us, "compute_time_us": pure_compute_us,
            "overlap_pct": 0,
        })
        data[label].append({
            "M": m, "K": k, "N": n, "pe_size": pe_size,
            "operator": "PTO", "mode": "fused",
            "time_us": pto_fused_us, "tflops": pto_tflops,
            "comm_bw_gbps": pto_bw,
            "comm_time_us": 0, "compute_time_us": 0,
            "overlap_pct": pto_overlap,
        })
        data[label].append({
            "M": m, "K": k, "N": n, "pe_size": pe_size,
            "operator": "SHMEM", "mode": "fused",
            "time_us": shmem_fused_us, "tflops": shmem_tflops,
            "comm_bw_gbps": shmem_bw,
            "comm_time_us": 0, "compute_time_us": 0,
            "overlap_pct": shmem_overlap,
        })

    return data


COLORS = {
    "comm": "#3498db",
    "compute": "#e74c3c",
    "pto_fused": "#2ecc71",
    "shmem_fused": "#f39c12",
    "pto_bar": "#27ae60",
    "shmem_bar": "#e67e22",
}


def format_time(us):
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.1f} µs"


@dataclass
class ChartContext:
    """Shared context for per-shape chart plotting."""
    label: str
    shape_title: str
    seq_entry: Optional[dict]
    pto_fused_entry: dict
    shmem_fused_entry: Optional[dict]
    output_dir: str


def _plot_time_chart(ctx: ChartContext):
    """Chart 1: Time Comparison (stacked bar for sequential, solid for fused)."""
    label, shape_title = ctx.label, ctx.shape_title
    seq_entry, pto_fused_entry, shmem_fused_entry = ctx.seq_entry, ctx.pto_fused_entry, ctx.shmem_fused_entry
    output_dir = ctx.output_dir
    fig, ax = plt.subplots(figsize=(9, 6))

    bar_labels = []
    bar_positions = []
    pos = 0

    if seq_entry:
        comm_t = seq_entry["comm_time_us"]
        comp_t = seq_entry["compute_time_us"]
        ax.bar(pos, comp_t, width=0.6, color=COLORS["compute"],
               label="Compute Time", edgecolor="white", linewidth=0.5)
        ax.bar(pos, comm_t, width=0.6, bottom=comp_t, color=COLORS["comm"],
               label="Comm Time", edgecolor="white", linewidth=0.5)
        total = comm_t + comp_t
        ax.text(pos, total + total * 0.02, format_time(total),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(pos, comp_t / 2, format_time(comp_t),
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(pos, comp_t + comm_t / 2, format_time(comm_t),
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bar_labels.append("Sequential\n(PTO)")
        bar_positions.append(pos)
        pos += 1

    if pto_fused_entry:
        t = pto_fused_entry["time_us"]
        ax.bar(pos, t, width=0.6, color=COLORS["pto_fused"],
               label="PTO Fused", edgecolor="white", linewidth=0.5)
        ax.text(pos, t + t * 0.02, format_time(t),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        bar_labels.append("Fused\n(PTO)")
        bar_positions.append(pos)
        pos += 1

    if shmem_fused_entry:
        t = shmem_fused_entry["time_us"]
        ax.bar(pos, t, width=0.6, color=COLORS["shmem_fused"],
               label="SHMEM Fused", edgecolor="white", linewidth=0.5)
        ax.text(pos, t + t * 0.02, format_time(t),
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        bar_labels.append("Fused\n(SHMEM)")
        bar_positions.append(pos)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("Time (µs)", fontsize=12)
    ax.set_title(f"Execution Time Comparison\n{shape_title}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{label}_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overlap_chart(ctx: ChartContext):
    """Chart 2: Overlap Percentage."""
    label, shape_title = ctx.label, ctx.shape_title
    pto_fused_entry, shmem_fused_entry, output_dir = ctx.pto_fused_entry, ctx.shmem_fused_entry, ctx.output_dir
    fig, ax = plt.subplots(figsize=(6, 5))

    overlap_labels = []
    overlap_values = []
    overlap_colors = []

    if pto_fused_entry:
        overlap_labels.append("PTO")
        overlap_values.append(pto_fused_entry["overlap_pct"])
        overlap_colors.append(COLORS["pto_bar"])

    if shmem_fused_entry:
        overlap_labels.append("SHMEM")
        overlap_values.append(shmem_fused_entry["overlap_pct"])
        overlap_colors.append(COLORS["shmem_bar"])

    bars = ax.bar(range(len(overlap_labels)), overlap_values, width=0.5,
                  color=overlap_colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, overlap_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(overlap_labels)))
    ax.set_xticklabels(overlap_labels, fontsize=12)
    ax.set_ylabel("Overlap (%)", fontsize=12)
    ax.set_title(f"Communication-Computation Overlap\n{shape_title}", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(overlap_values + [50]) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{label}_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bandwidth_chart(ctx: ChartContext):
    """Chart 3: Bandwidth Comparison."""
    label, shape_title = ctx.label, ctx.shape_title
    pto_fused_entry, shmem_fused_entry, output_dir = ctx.pto_fused_entry, ctx.shmem_fused_entry, ctx.output_dir
    fig, ax = plt.subplots(figsize=(6, 5))

    bw_labels = []
    bw_values = []
    bw_colors = []

    if pto_fused_entry and pto_fused_entry["comm_bw_gbps"] > 0:
        bw_labels.append("PTO")
        bw_values.append(pto_fused_entry["comm_bw_gbps"])
        bw_colors.append(COLORS["pto_bar"])

    if shmem_fused_entry and shmem_fused_entry["comm_bw_gbps"] > 0:
        bw_labels.append("SHMEM")
        bw_values.append(shmem_fused_entry["comm_bw_gbps"])
        bw_colors.append(COLORS["shmem_bar"])

    if bw_values:
        bars = ax.bar(range(len(bw_labels)), bw_values, width=0.5,
                      color=bw_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, bw_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bw_values) * 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_xticks(range(len(bw_labels)))
        ax.set_xticklabels(bw_labels, fontsize=12)
    ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax.set_title(f"Effective Communication Bandwidth\n{shape_title}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{label}_bandwidth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shape(label, entries, output_dir):
    """Generate 3 charts for a single matrix shape."""
    seq_entry = None
    pto_fused_entry = None
    shmem_fused_entry = None

    for e in entries:
        if e["operator"] == "PTO" and e["mode"] == "sequential":
            seq_entry = e
        elif e["operator"] == "PTO" and e["mode"] == "fused":
            pto_fused_entry = e
        elif e["operator"] == "SHMEM" and e["mode"] == "fused":
            shmem_fused_entry = e

    if pto_fused_entry is None:
        logger.warning("No PTO fused data for %s, skipping", label)
        return

    m, k, n = pto_fused_entry["M"], pto_fused_entry["K"], pto_fused_entry["N"]
    pe_size = pto_fused_entry["pe_size"]
    shape_title = f"M={m}, K={k}, N={n} (PE={pe_size})"

    ctx = ChartContext(label, shape_title, seq_entry, pto_fused_entry, shmem_fused_entry, output_dir)
    _plot_time_chart(ctx)
    _plot_overlap_chart(ctx)
    _plot_bandwidth_chart(ctx)


def _collect_summary_data(data):
    """Collect PTO/SHMEM fused data across all shapes for summary chart."""
    pto_times = []
    shmem_times = []
    pto_overlaps = []
    shmem_overlaps = []
    shape_labels = []

    for label in data:
        entries = data[label]
        pto_f = next((e for e in entries if e["operator"] == "PTO" and e["mode"] == "fused"), None)
        shmem_f = next((e for e in entries if e["operator"] == "SHMEM" and e["mode"] == "fused"), None)

        if pto_f:
            shape_labels.append(label)
            pto_times.append(pto_f["time_us"])
            pto_overlaps.append(pto_f["overlap_pct"])
            shmem_times.append(shmem_f["time_us"] if shmem_f else 0)
            shmem_overlaps.append(shmem_f["overlap_pct"] if shmem_f else 0)

    return shape_labels, pto_times, shmem_times, pto_overlaps, shmem_overlaps


@dataclass
class SummarySubplotConfig:
    """Configuration for one subplot in the summary chart."""
    x: np.ndarray
    width: float
    pto_vals: list
    shmem_vals: list
    shape_labels: list
    ylabel: str
    title: str
    y_fmt: object = None


def _format_summary_subplot(ax, cfg: SummarySubplotConfig):
    """Configure one subplot in the summary chart."""
    ax.bar(cfg.x - cfg.width / 2, cfg.pto_vals, cfg.width,
           color=COLORS["pto_bar"], label="PTO", edgecolor="white")
    if any(v > 0 for v in cfg.shmem_vals):
        ax.bar(cfg.x + cfg.width / 2, cfg.shmem_vals, cfg.width,
               color=COLORS["shmem_bar"], label="SHMEM", edgecolor="white")
    ax.set_xlabel("Matrix Shape (MxKxN)", fontsize=11)
    ax.set_ylabel(cfg.ylabel, fontsize=11)
    ax.set_title(cfg.title, fontsize=13, fontweight="bold")
    ax.set_xticks(cfg.x)
    ax.set_xticklabels(cfg.shape_labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    if cfg.y_fmt:
        ax.yaxis.set_major_formatter(cfg.y_fmt)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_summary(data, output_dir):
    """Generate a summary chart across all shapes."""
    if len(data) < 2:
        return

    shape_labels, pto_times, shmem_times, pto_overlaps, shmem_overlaps = _collect_summary_data(data)
    if not shape_labels:
        return

    x = np.arange(len(shape_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    _format_summary_subplot(ax1, SummarySubplotConfig(
        x, width, pto_times, shmem_times, shape_labels,
        "Fused Time (µs)", "Fused Execution Time Across Shapes",
        ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")))
    _format_summary_subplot(ax2, SummarySubplotConfig(
        x, width, pto_overlaps, shmem_overlaps, shape_labels,
        "Overlap (%)", "Comm-Compute Overlap Across Shapes"))
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot AllGather-GEMM benchmark results"
    )
    parser.add_argument("csv_path", nargs="?", default=None,
                        help="Path to benchmark_data.csv (omit for sample data demo)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory for charts")
    parser.add_argument("--sample", action="store_true",
                        help="Use generated sample data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.csv_path and not args.sample:
        if not os.path.isfile(args.csv_path):
            logger.error("CSV file not found: %s", args.csv_path)
            sys.exit(1)
        data = parse_csv(args.csv_path)
        default_output = os.path.join(os.path.dirname(args.csv_path), "charts")
    else:
        logger.info("Using sample data for demonstration")
        data = generate_sample_data()
        default_output = "./benchmark_charts_sample"

    output_dir = args.output_dir or default_output
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating charts for %d shapes...", len(data))
    logger.info("Output directory: %s", output_dir)

    for label, entries in data.items():
        logger.info("  Plotting %s...", label)
        plot_shape(label, entries, output_dir)

    plot_summary(data, output_dir)

    logger.info("Charts saved to %s/", output_dir)
    logger.info("  Per-shape charts:")
    for label in data:
        logger.info("    %s_time.png", label)
        logger.info("    %s_overlap.png", label)
        logger.info("    %s_bandwidth.png", label)
    if len(data) >= 2:
        logger.info("  Summary chart: summary.png")


if __name__ == "__main__":
    main()
