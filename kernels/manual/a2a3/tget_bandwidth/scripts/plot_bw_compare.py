#!/usr/bin/env python3
"""
Plot TGET vs TGET_ASYNC bandwidth trends.

Host-side bandwidth comes from the latest benchmark output.
Device-side bandwidth follows the repo convention used in flash attention:
  us = cycles * 20 / 1000
which implies:
  GB/s = bytes / us / 1e3
"""

from pathlib import Path

import matplotlib.pyplot as plt


BYTES = [4096, 16384, 65536, 262144, 1048576, 4194304]
SIZE_LABELS = ["4KB", "16KB", "64KB", "256KB", "1MB", "4MB"]

HOST_BW = {
    "TGET": [0.21, 0.72, 1.75, 3.01, 3.75, 3.99],
    "TGET_ASYNC": [0.19, 0.75, 2.55, 6.08, 10.48, 12.95],
}

DEVICE_AVG_CYCLES = {
    "TGET": [50.85, 202.05, 780.73, 3347.12, 12703.39, 52878.12],
    "TGET_ASYNC": [118.18, 166.42, 338.10, 1094.37, 3791.18, 14834.47],
}


def cycles_to_bandwidth_gbps(nbytes: int, avg_cycles: float) -> float:
    avg_us = avg_cycles * 20.0 / 1000.0
    return nbytes / avg_us / 1e3


DEVICE_BW = {
    name: [cycles_to_bandwidth_gbps(nbytes, cycles) for nbytes, cycles in zip(BYTES, values)]
    for name, values in DEVICE_AVG_CYCLES.items()
}


def plot_subplot(ax, title, series):
    colors = {"TGET": "#1f77b4", "TGET_ASYNC": "#d62728"}
    markers = {"TGET": "o", "TGET_ASYNC": "s"}
    for name, values in series.items():
        ax.plot(
            SIZE_LABELS,
            values,
            label=name,
            color=colors[name],
            marker=markers[name],
            linewidth=2.2,
            markersize=7,
        )
    ax.set_title(title)
    ax.set_xlabel("Transfer Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()


def main():
    output_path = Path(__file__).resolve().parent.parent / "tget_bw_compare.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    plot_subplot(axes[0], "Host-side Measured Bandwidth", HOST_BW)
    plot_subplot(axes[1], "Device-side Bandwidth (20ns/tick)", DEVICE_BW)

    fig.suptitle("TGET vs TGET_ASYNC Bandwidth Trend on A2/A3", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"saved plot to: {output_path}")


if __name__ == "__main__":
    main()
