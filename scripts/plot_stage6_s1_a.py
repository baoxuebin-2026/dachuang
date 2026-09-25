#!/usr/bin/env python3
"""Figures from completed Stage 6 result tables, without refitting or replaying.

python scripts/plot_stage6_s1_a.py --input results/stage6_s1_a --out figures/stage6_s1_a
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, dest: Path) -> None:
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
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})

    traces = pd.read_csv(args.input / "illustrative_timeline.csv")
    chosen = sorted(traces.trial.unique())[0]  # declared lexicographically first trial
    t = traces[(traces.trial == chosen) & (traces.scenario == "S2_possible_overlap")].sort_values(
        "completed_time_s")
    if len(t) != 215:
        raise ValueError("Missing illustrative second-long windows")
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 4.9), sharex=True)
    specs = (("gas_evidence", "Gas (lab): G", "#245a79"),
             ("person_relation", "Person (sim): P", "#d07838"),
             ("joint_level", "Handling (sim): L", "#198678"))
    for ax, (column, ylabel, color) in zip(axs, specs):
        if column == "joint_level":
            values = t[column].to_numpy()
        else:
            values = t[column].str.slice(1).astype(int).to_numpy()
        ax.step(t.completed_time_s.to_numpy(), values, where="post", color=color, lw=1.4)
        ax.axvline(60, linestyle="--", color="#777777", alpha=0.75, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_yticks(range(4 if column == "joint_level" else 3))
        ax.grid(axis="y", alpha=0.2)
    axs[-1].set_xlabel("Laboratory relative time / simulated pairing time (s)")
    axs[-1].set_xlim(25, 240)
    save(fig, args.out / "illustrative_s2_timeline")

    sensitivity = pd.read_csv(args.input / "sensitivity_aggregate.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 3.1), sharey=True)
    shades = {0.0: "#245a79", 0.2: "#198678", 0.5: "#c77940"}
    for ax, duration in zip(axes, (1, 3, 5)):
        block = sensitivity[sensitivity.duration_s == duration]
        for bound in (0.0, 0.2, 0.5):
            current = block[block.position_error_bound_m == bound].sort_values("buffer_m")
            ax.plot(current.buffer_m, current.total_L3_seconds_across_replays / current.trials,
                    marker="o", color=shades[bound], label=f"error bound {bound:g} m")
        ax.set_title(f"Soft evidence {duration} s", fontsize=9)
        ax.set_xticks([0.5, 1, 2])
        ax.set_xlabel("Simulated buffer (m)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mean L3 seconds per replayed file")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 1.09))
    save(fig, args.out / "s2_parameter_sensitivity")


if __name__ == "__main__":
    main()
