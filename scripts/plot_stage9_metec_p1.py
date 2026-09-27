"""Plot the formal METEC P1 per-date result tables without recomputing metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
        }
    )


def save_threshold_figure(rows: list[dict[str, str]], out_dir: Path) -> None:
    dates = [row["held_out_date"][5:] for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True, constrained_layout=True)
    for axis, quantile, label in zip(axes, ("p95", "p99"), ("P95", "P99")):
        release = [100 * float(row[f"{quantile}_release_fraction"]) for row in rows]
        outside = [100 * float(row[f"{quantile}_outside_fraction"]) for row in rows]
        axis.plot(x, release, color="#2F6B9A", linewidth=1.2, marker="o", markersize=2.5, label="释放观测秒")
        axis.plot(x, outside, color="#C65D3B", linewidth=1.0, marker="s", markersize=2.2, label="无记录释放观测秒")
        axis.set_ylabel("超阈值比例（%）")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.text(0.01, 0.92, label, transform=axis.transAxes, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    step = 2
    axes[-1].set_xticks(x[::step], dates[::step], rotation=60, ha="right")
    axes[-1].set_xlabel("留出日期（2024年）")
    for extension, dpi in (("svg", 300), ("png", 600)):
        fig.savefig(out_dir / f"per_date_threshold_tradeoff.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_auc_figure(rows: list[dict[str, str]], out_dir: Path) -> None:
    dates = [row["held_out_date"][5:] for row in rows]
    auc = [float(row["rank_auc"]) for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    axis.plot(x, auc, color="#446E5C", linewidth=1.1, marker="o", markersize=3)
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.9, label="随机排序 0.5")
    axis.set_ylim(0, 1)
    axis.set_ylabel("日期内 AUC")
    axis.set_xlabel("留出日期（2024年）")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    axis.legend(loc="lower right", frameon=False)
    step = 2
    axis.set_xticks(x[::step], dates[::step], rotation=60, ha="right")
    for extension, dpi in (("svg", 300), ("png", 600)):
        fig.savefig(out_dir / f"per_date_auc.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-date", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.per_date)
    if not rows:
        raise ValueError("per-date result table is empty")
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    save_threshold_figure(rows, args.out)
    save_auc_figure(rows, args.out)


if __name__ == "__main__":
    main()
