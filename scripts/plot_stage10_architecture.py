#!/usr/bin/env python3
"""Draw the manuscript architecture/evidence-boundary figure.

The figure deliberately separates the proposed deployment from the evidence
that was actually evaluated.  It is a design/provenance figure, not a system
performance figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "stage10_architecture"


COLORS = {
    "navy": "#173B57",
    "blue": "#2F6F9F",
    "blue_fill": "#EAF3F8",
    "teal": "#2B7A78",
    "teal_fill": "#E7F4F1",
    "gold": "#A56A00",
    "gold_fill": "#FFF4D6",
    "red": "#9B3D34",
    "red_fill": "#FBECE9",
    "gray": "#5D6972",
    "gray_fill": "#F2F4F5",
    "ink": "#1E2B32",
    "white": "#FFFFFF",
}


def box(ax, xy, wh, title, body, *, edge, face, dashed=False, title_size=9.2, body_size=7.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.35,
        edgecolor=edge,
        facecolor=face,
        linestyle=(0, (4, 2)) if dashed else "solid",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", color=COLORS["ink"], zorder=3)
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=body_size,
            color=COLORS["ink"], linespacing=1.28, zorder=3)
    return patch


def arrow(ax, start, end, *, color=None, dashed=False, width=1.4, zorder=1):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=width,
        color=color or COLORS["gray"],
        linestyle=(0, (4, 2)) if dashed else "solid",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def panel(ax, xy, wh, title, subtitle, *, color):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.6,
            edgecolor=color,
            facecolor=COLORS["white"],
            linestyle=(0, (5, 2)),
            zorder=0,
        )
    )
    ax.text(x + 0.018, y + h - 0.038, title, ha="left", va="center", fontsize=11.2,
            fontweight="bold", color=color)
    ax.text(x + w - 0.018, y + h - 0.038, subtitle, ha="right", va="center", fontsize=7.4,
            color=COLORS["gray"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.0, 7.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.972, "Proposed deployment and actual validation boundary", ha="center", va="top",
            fontsize=14.2, fontweight="bold", color=COLORS["navy"])
    ax.text(0.5, 0.942, "Design mapping for the coking-park warning study", ha="center", va="top",
            fontsize=8.6, color=COLORS["gray"])

    # Proposed system: everything in this panel remains prospective.
    panel(ax, (0.035, 0.535), (0.93, 0.36), "(a) Proposed deployment architecture",
          "No D435, field sensor or Jetson measurement in this study", color=COLORS["blue"])

    box(ax, (0.065, 0.595), (0.17, 0.15), "RGB-D sensing", "D435-class camera\nRGB + registered depth",
        edge=COLORS["blue"], face=COLORS["blue_fill"], dashed=True)
    box(ax, (0.065, 0.765), (0.17, 0.065), "PROPOSED", "device mapping", edge=COLORS["red"],
        face=COLORS["red_fill"], dashed=True, title_size=8.2, body_size=6.4)

    box(ax, (0.285, 0.595), (0.17, 0.15), "Gas / environment", "Gas concentration, T/RH\nand task-specific sensors",
        edge=COLORS["blue"], face=COLORS["blue_fill"], dashed=True)
    box(ax, (0.285, 0.765), (0.17, 0.065), "PROPOSED", "field instruments", edge=COLORS["red"],
        face=COLORS["red_fill"], dashed=True, title_size=8.2, body_size=6.4)

    box(ax, (0.505, 0.595), (0.20, 0.235), "Evidence contract",
        "source + timestamp\nzone + coordinates\nunit + quality + age\nmissing reason",
        edge=COLORS["teal"], face=COLORS["teal_fill"], dashed=True)
    box(ax, (0.755, 0.595), (0.18, 0.235), "Edge decision node",
        "Jetson-class target\nmodule inference\ntime/zone/quality gates\nL0-L3 + review log",
        edge=COLORS["teal"], face=COLORS["teal_fill"], dashed=True)

    arrow(ax, (0.235, 0.67), (0.505, 0.70), color=COLORS["blue"], dashed=True)
    arrow(ax, (0.455, 0.67), (0.505, 0.67), color=COLORS["blue"], dashed=True)
    arrow(ax, (0.705, 0.712), (0.755, 0.712), color=COLORS["teal"], dashed=True)

    # Actual evidence: four isolated validation paths.  There is intentionally
    # no Bonn-to-S1 arrow because those datasets were never synchronized.
    panel(ax, (0.035, 0.085), (0.93, 0.39), "(b) Evidence actually evaluated",
          "Module experiments and simulation remain separate", color=COLORS["gray"])

    xs = [0.065, 0.285, 0.505, 0.725]
    widths = [0.17, 0.17, 0.17, 0.21]
    sources = [
        ("UCI 309", "wind-tunnel files\n8 gas channels", COLORS["gold"], COLORS["gold_fill"]),
        ("Bonn RGB-D", "two indoor sequences\nteam 2D labels", COLORS["gold"], COLORS["gold_fill"]),
        ("UCI 309 + SIM", "same 60 gas files\none simulated path", COLORS["red"], COLORS["red_fill"]),
        ("METEC", "controlled releases\nsparse point CH4", COLORS["gold"], COLORS["gold_fill"]),
    ]
    outputs = [
        ("Gas evidence", "release-response\nfile metrics"),
        ("Visual interface", "2D boxes + depth\nquality / virtual zone"),
        ("S1 rule replay", "fault injection +\ngate comparison"),
        ("Threshold diagnostic", "leave-one-date-out\nfixed P95 / P99"),
    ]
    for x, w, (st, sb, edge, face), (ot, ob) in zip(xs, widths, sources, outputs):
        box(ax, (x, 0.285), (w, 0.105), st, sb, edge=edge, face=face, title_size=8.8, body_size=7.0)
        box(ax, (x, 0.135), (w, 0.105), ot, ob, edge=COLORS["gray"], face=COLORS["gray_fill"],
            title_size=8.5, body_size=6.9)
        arrow(ax, (x + w / 2, 0.285), (x + w / 2, 0.24), color=edge, width=1.35)

    ax.text(0.5, 0.100,
            "Supported claim: reproducible module behavior and auditable evidence handling under stated conditions",
            ha="center", va="center", fontsize=7.8, color=COLORS["navy"], fontweight="bold")
    ax.text(0.5, 0.045,
            "No same-site, same-clock visual-gas observations; no field warning accuracy or hardware performance claim",
            ha="center", va="center", fontsize=8.1, color=COLORS["red"], fontweight="bold")

    fig.savefig(OUT / "proposed_vs_evidence_boundary.svg", bbox_inches="tight")
    fig.savefig(OUT / "proposed_vs_evidence_boundary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
