#!/usr/bin/env python3
"""Plot UCI 487 diagnostics and frozen V3 result tables; never refit models.

python scripts/plot_uci487_v3.py --input results/uci487_v3 --out figures/uci487_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export(fig, dest: Path) -> None:
    fig.tight_layout()
    fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(dest.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.input / "v1_daily.csv")
    scores = pd.read_csv(args.input / "test_by_day.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ax.plot(np.arange(1, 14), daily.zero_co_r1_low_mohm,
            color="#176585", marker="o", linewidth=1.5, label="R1, low heater")
    ax.set_xlabel("Measurement day (chronological index)")
    ax.set_ylabel("Median R1 resistance at 0 ppm CO (MOhm)")
    ax.set_xticks(np.arange(1, 14))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    export(fig, args.out / "v1_zero_co_r1_day")

    labels = {"constant_train_median": "Train median CO",
              "ambient_only": "Humidity + temperature",
              "r1_blind": "R1 only",
              "all14_blind": "14 sensors, heater ignored",
              "all14_phase": "14 sensors, heater separated",
              "all14_phase_ambient": "14 sensors + heater + ambient"}
    colors = {"constant_train_median": "#777777", "ambient_only": "#a17758",
              "r1_blind": "#d08449", "all14_blind": "#82a6ae",
              "all14_phase": "#145878", "all14_phase_ambient": "#178f7f"}
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    days = sorted(scores.day.unique())
    for model in labels:
        block = scores.loc[scores.model == model].set_index("day").loc[days]
        ax.plot(np.arange(1, 4), block.mae_ppm, color=colors[model], marker="o",
                linewidth=1.5, label=labels[model])
    ax.set_xlabel("Held-out day (chronological order)")
    ax.set_xticks(np.arange(1, 4))
    ax.set_ylabel("Completed-segment CO MAE (ppm)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    export(fig, args.out / "v3_test_mae_by_day")


if __name__ == "__main__":
    main()
