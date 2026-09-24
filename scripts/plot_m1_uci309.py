#!/usr/bin/env python3
"""Plot *existing* UCI 309 M1 results; performs no model computations.

python scripts/plot_m1_uci309.py --results results/m1_uci309 \
    --out figures/m1_uci309
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = pd.read_csv(args.results / "summary.csv")
    metadata = json.loads((args.results / "run.json").read_text(encoding="utf-8"))
    names = ["Single sensor", "Eight-sensor median", "EWMA", "CUSUM",
             "Temp + humidity", "Program clock (invalid)"]
    selections = [("gas", "single"), ("gas", "multi"), ("gas", "ewma"),
                  ("gas", "cusum"), ("ambient", "multi")]
    records = []
    for source, method in selections:
        rows = summary.loc[(summary["mode"] == "A") & (summary["source"] == source)
                           & (summary["method"] == method) & (summary["split"] == "test")
                           & (summary["control"] == "real_laboratory_data")]
        if len(rows) != 1 or int(rows.iloc[0].n_files) != 60:
            raise ValueError(f"Missing or repeated blinded test result: {source}/{method}")
        records.append(rows.iloc[0])
    oracle = metadata["clock_oracle"]
    y = list(range(len(names)))
    colors = ["#4E79A7"] * 4 + ["#7F7F7F", "#F28E2B"]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True,
                            gridspec_kw={"width_ratios": [1.15, 0.9]})
    for ax, values, xlabel, xmax in (
        (axs[0], [int(r.detected_within_60_s) for r in records] +
         [int(oracle["detected"])], "Detected within 60 s of release (out of 60)", 64),
        (axs[1], [int(r.false_files) for r in records] +
         [int(oracle["false_files"])], "Trials with pre-release trigger (out of 60)", 11),
    ):
        bars = ax.barh(y, values, color=colors, height=0.65)
        bars[-1].set_hatch("///")
        for pos, value in zip(y, values):
            ax.text(value + xmax * 0.017, pos, str(value), va="center", fontsize=9)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(y, names)
    axs[0].invert_yaxis()
    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.96, wspace=0.11)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "m1_test_comparison.svg")
    fig.savefig(args.out / "m1_test_comparison.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
