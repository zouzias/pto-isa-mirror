#!/usr/bin/env python3
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
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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
        print(f"[WARN] No PTO fused data for {label}, skipping")
        return

    m, k, n = pto_fused_entry["M"], pto_fused_entry["K"], pto_fused_entry["N"]
    pe_size = pto_fused_entry["pe_size"]
    shape_title = f"M={m}, K={k}, N={n} (PE={pe_size})"

    # ========================================================================
    # Chart 1: Time Comparison (stacked bar for sequential, solid for fused)
    # ========================================================================
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    bar_labels = []
    bar_positions = []
    pos = 0

    if seq_entry:
        comm_t = seq_entry["comm_time_us"]
        comp_t = seq_entry["compute_time_us"]
        b1 = ax1.bar(pos, comp_t, width=0.6, color=COLORS["compute"],
                     label="Compute Time", edgecolor="white", linewidth=0.5)
        b2 = ax1.bar(pos, comm_t, width=0.6, bottom=comp_t, color=COLORS["comm"],
                     label="Comm Time", edgecolor="white", linewidth=0.5)
        total = comm_t + comp_t
        ax1.text(pos, total + total * 0.02, format_time(total),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax1.text(pos, comp_t / 2, format_time(comp_t),
                 ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax1.text(pos, comp_t + comm_t / 2, format_time(comm_t),
                 ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bar_labels.append("Sequential\n(PTO)")
        bar_positions.append(pos)
        pos += 1

    if pto_fused_entry:
        t = pto_fused_entry["time_us"]
        ax1.bar(pos, t, width=0.6, color=COLORS["pto_fused"],
                label="PTO Fused", edgecolor="white", linewidth=0.5)
        ax1.text(pos, t + t * 0.02, format_time(t),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        bar_labels.append("Fused\n(PTO)")
        bar_positions.append(pos)
        pos += 1

    if shmem_fused_entry:
        t = shmem_fused_entry["time_us"]
        ax1.bar(pos, t, width=0.6, color=COLORS["shmem_fused"],
                label="SHMEM Fused", edgecolor="white", linewidth=0.5)
        ax1.text(pos, t + t * 0.02, format_time(t),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
        bar_labels.append("Fused\n(SHMEM)")
        bar_positions.append(pos)
        pos += 1

    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(bar_labels, fontsize=11)
    ax1.set_ylabel("Time (µs)", fontsize=12)
    ax1.set_title(f"Execution Time Comparison\n{shape_title}", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"{label}_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ========================================================================
    # Chart 2: Overlap Percentage
    # ========================================================================
    fig2, ax2 = plt.subplots(figsize=(6, 5))

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

    bars = ax2.bar(range(len(overlap_labels)), overlap_values, width=0.5,
                   color=overlap_colors, edgecolor="white", linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, overlap_values)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax2.set_xticks(range(len(overlap_labels)))
    ax2.set_xticklabels(overlap_labels, fontsize=12)
    ax2.set_ylabel("Overlap (%)", fontsize=12)
    ax2.set_title(f"Communication-Computation Overlap\n{shape_title}", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(overlap_values + [50]) * 1.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"{label}_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ========================================================================
    # Chart 3: Bandwidth Comparison
    # ========================================================================
    fig3, ax3 = plt.subplots(figsize=(6, 5))

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
        bars = ax3.bar(range(len(bw_labels)), bw_values, width=0.5,
                       color=bw_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, bw_values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bw_values) * 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax3.set_xticks(range(len(bw_labels)))
        ax3.set_xticklabels(bw_labels, fontsize=12)
    ax3.set_ylabel("Bandwidth (GB/s)", fontsize=12)
    ax3.set_title(f"Effective Communication Bandwidth\n{shape_title}", fontsize=13, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, f"{label}_bandwidth.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)


def plot_summary(data, output_dir):
    """Generate a summary chart across all shapes."""
    labels = list(data.keys())
    if len(labels) < 2:
        return

    pto_times = []
    shmem_times = []
    pto_overlaps = []
    shmem_overlaps = []
    shape_labels = []

    for label in labels:
        entries = data[label]
        pto_f = next((e for e in entries if e["operator"] == "PTO" and e["mode"] == "fused"), None)
        shmem_f = next((e for e in entries if e["operator"] == "SHMEM" and e["mode"] == "fused"), None)

        if pto_f:
            shape_labels.append(label)
            pto_times.append(pto_f["time_us"])
            pto_overlaps.append(pto_f["overlap_pct"])
            shmem_times.append(shmem_f["time_us"] if shmem_f else 0)
            shmem_overlaps.append(shmem_f["overlap_pct"] if shmem_f else 0)

    if not shape_labels:
        return

    x = np.arange(len(shape_labels))
    width = 0.35

    # Summary: Fused time comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    bars1 = ax1.bar(x - width / 2, pto_times, width, color=COLORS["pto_bar"],
                    label="PTO", edgecolor="white")
    if any(t > 0 for t in shmem_times):
        bars2 = ax1.bar(x + width / 2, shmem_times, width, color=COLORS["shmem_bar"],
                        label="SHMEM", edgecolor="white")

    ax1.set_xlabel("Matrix Shape (MxKxN)", fontsize=11)
    ax1.set_ylabel("Fused Time (µs)", fontsize=11)
    ax1.set_title("Fused Execution Time Across Shapes", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(shape_labels, rotation=30, ha="right", fontsize=9)
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    # Summary: Overlap comparison
    bars3 = ax2.bar(x - width / 2, pto_overlaps, width, color=COLORS["pto_bar"],
                    label="PTO", edgecolor="white")
    if any(o > 0 for o in shmem_overlaps):
        bars4 = ax2.bar(x + width / 2, shmem_overlaps, width, color=COLORS["shmem_bar"],
                        label="SHMEM", edgecolor="white")

    ax2.set_xlabel("Matrix Shape (MxKxN)", fontsize=11)
    ax2.set_ylabel("Overlap (%)", fontsize=11)
    ax2.set_title("Comm-Compute Overlap Across Shapes", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(shape_labels, rotation=30, ha="right", fontsize=9)
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

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

    if args.csv_path and not args.sample:
        if not os.path.isfile(args.csv_path):
            print(f"[ERROR] CSV file not found: {args.csv_path}")
            sys.exit(1)
        data = parse_csv(args.csv_path)
        default_output = os.path.join(os.path.dirname(args.csv_path), "charts")
    else:
        print("[INFO] Using sample data for demonstration")
        data = generate_sample_data()
        default_output = "./benchmark_charts_sample"

    output_dir = args.output_dir or default_output
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Generating charts for {len(data)} shapes...")
    print(f"[INFO] Output directory: {output_dir}")

    for label, entries in data.items():
        print(f"  - Plotting {label}...")
        plot_shape(label, entries, output_dir)

    plot_summary(data, output_dir)

    print(f"\n[DONE] Charts saved to {output_dir}/")
    print("  Per-shape charts:")
    for label in data:
        print(f"    {label}_time.png")
        print(f"    {label}_overlap.png")
        print(f"    {label}_bandwidth.png")
    if len(data) >= 2:
        print("  Summary chart: summary.png")


if __name__ == "__main__":
    main()
