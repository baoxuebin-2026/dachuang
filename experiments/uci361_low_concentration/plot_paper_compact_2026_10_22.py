"""Publication figures from verified exposure-level predictions.

Each CSV row is one physical gas exposure. No test record is resampled, and
the three concentration ranks pool four gases by ordinal rank only.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "论文结果图_精简版_2026-10-22"
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont(str(HERE / "assets/NotoSansSC-figure-subset.otf"))
plt.rcParams.update({
    "font.family": font_manager.FontProperties(
        fname=str(HERE / "assets/NotoSansSC-figure-subset.otf")
    ).get_name(),
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=600,
                    bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


pred = pd.read_csv(HERE / "uci361_recovered_predictions_2026-10-20.csv")
pred = pred.loc[pred.split.eq("one")].copy()
assert len(pred) == pred.file.nunique() == 192
pred["rank"] = pred.file.str.extract(r"_F0([123])0_").astype(int)
assert pred.groupby(["rank", "truth"]).size().eq(16).all()
columns = ["raw_cnn", "norm_cnn", "aug_cnn"]
assert [(pred.truth != pred[c]).sum() for c in columns] == [60, 8, 5]

# Fig. 1: one point = incorrect exposures out of 64 at one ordinal rank.
# Separate, explicitly labeled axes reveal differences near zero without
# hiding the much larger unaugmented error counts or drawing a broken axis.
fig, axes = plt.subplots(2, 1, figsize=(5.55, 3.25), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.04], "hspace": 0.23})
counts = {
    column: [int((part.truth != part[column]).sum())
             for _, part in pred.groupby("rank")]
    for column in columns
}
assert counts == {"raw_cnn": [38, 18, 4],
                  "norm_cnn": [1, 4, 3], "aug_cnn": [4, 1, 0]}
upper, lower = axes
upper.plot([1, 2, 3], counts["raw_cnn"], marker="o", markersize=5.2,
           linewidth=1.6, color="#505B66", label="原始 CNN")
for label, column, color, marker in (
    ("逐试次归一化 CNN", "norm_cnn", "#007F7A", "s"),
    ("幅值增强 CNN", "aug_cnn", "#AF4B37", "^"),
):
    lower.plot([1, 2, 3], counts[column], marker=marker, markersize=5.2,
               linewidth=1.6, color=color, label=label)
upper.set_ylim(0, 42)
lower.set_ylim(-0.45, 8)
upper.set_yticks((0, 10, 20, 30, 40))
lower.set_yticks((0, 2, 4, 6, 8))
lower.set_xticks((1, 2, 3), ["第1档", "第2档", "第3档"])
for ax in axes:
    ax.set_xlim(0.85, 3.15)
    ax.grid(axis="y", color="#D5DADF", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2,
              handlelength=1.9, columnspacing=0.8)
fig.supylabel("误判供气试次/次", x=0.02, fontsize=9)
fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.98)
save(fig, "fig1_low_rank_errors")

# Fig. 2: original confusion counts, with a shared scale for both methods.
codes = ["CO", "Me", "Ea", "Ey"]
names = ["CO", "甲烷", "乙醇", "乙烯"]
matrices = [confusion_matrix(pred.truth, pred[column], labels=codes)
            for column in ("raw_cnn", "aug_cnn")]
assert [int(np.trace(matrix)) for matrix in matrices] == [132, 187]
fig, axes = plt.subplots(1, 2, figsize=(6.05, 2.85), sharex=True, sharey=True)
for ax, matrix, title in zip(axes, matrices,
                             ("（a）原始 CNN", "（b）幅值增强 CNN")):
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=48,
                   interpolation="none", aspect="equal")
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("预测类别")
    ax.set_title(title, fontsize=9, pad=6)
    for (i, j), value in np.ndenumerate(matrix):
        if value:
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.5,
                    color="white" if value >= 25 else "#263440")
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
axes[0].set_ylabel("真实类别")
fig.subplots_adjust(left=0.10, right=0.89, bottom=0.15, top=0.87, wspace=0.13)
colorbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.022,
                        ticks=(0, 12, 24, 36, 48))
colorbar.set_label("供气试次/次", fontsize=8)
save(fig, "fig2_confusion_matrices")
print(f"Wrote four PNG/SVG files in {OUT}")
