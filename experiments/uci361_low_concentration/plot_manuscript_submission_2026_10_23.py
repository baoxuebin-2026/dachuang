"""Generate publication-grade PNG/SVG figures for the revised manuscript.

All values are loaded from verified exposure-level predictions or frozen
experiment outputs. The script creates no synthetic observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, patches
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "论文结果图_投稿版_2026-10-23"
OUT.mkdir(exist_ok=True)
FONT = HERE / "assets/NotoSansSC-figure-subset.otf"
font_manager.fontManager.addfont(str(FONT))
FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.4,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

NAVY = "#275D7A"
ORANGE = "#C65D3B"
TEAL = "#22847A"
INK = "#24323D"
MID = "#6D7882"
GRID = "#D7DDE2"
PALE_BLUE = "#EAF2F6"
PALE_ORANGE = "#F8EEE9"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, face, edge, fontsize=8.5):
    box = patches.FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.0, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK,
            linespacing=1.3)
    return box


def arrow(ax, start, end, color=MID):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", lw=1.05, color=color,
                                mutation_scale=10, shrinkA=2, shrinkB=2))


# Figure 1: task and method pipeline. The target branch is visually separated
# from source-only training so the absence of target labels is unambiguous.
fig, ax = plt.subplots(figsize=(6.9, 2.25))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.add_patch(patches.FancyBboxPatch(
    (0.015, 0.51), 0.715, 0.43,
    boxstyle="round,pad=0.012,rounding_size=0.025",
    facecolor="#F7F9FA", edgecolor="#B6C0C8", linewidth=0.8,
))
ax.text(0.035, 0.90, "训练部分：仅使用源设备", fontsize=8.2, color=NAVY,
        va="center")
rounded_box(ax, (0.035, 0.60), 0.16, 0.20, "源设备高浓度\n第4—10档", PALE_BLUE, NAVY)
rounded_box(ax, (0.245, 0.60), 0.16, 0.20, "8 MOX 对数值\n后值 - 前值", "#F1F3F5", MID)
rounded_box(ax, (0.455, 0.60), 0.19, 0.20, "3档幅值增强\n0.50 / 0.75 / 1.00", PALE_ORANGE, ORANGE)
rounded_box(ax, (0.690, 0.60), 0.15, 0.20, "SVM + CNN\n四类模型训练", PALE_BLUE, NAVY)
arrow(ax, (0.195, 0.70), (0.245, 0.70))
arrow(ax, (0.405, 0.70), (0.455, 0.70))
arrow(ax, (0.645, 0.70), (0.690, 0.70))

ax.add_patch(patches.FancyBboxPatch(
    (0.015, 0.08), 0.97, 0.31,
    boxstyle="round,pad=0.012,rounding_size=0.025",
    facecolor="#FBFCFC", edgecolor="#B6C0C8", linewidth=0.8,
))
ax.text(0.035, 0.34, "测试部分：目标设备保持留出", fontsize=8.2, color=TEAL,
        va="center")
rounded_box(ax, (0.035, 0.13), 0.19, 0.15, "目标设备低浓度\n第1—3档", "#EAF5F3", TEAL)
rounded_box(ax, (0.315, 0.13), 0.17, 0.15, "已训练模型\n直接预测", PALE_BLUE, NAVY)
rounded_box(ax, (0.575, 0.13), 0.19, 0.15, "逐文件计分\n四类、CO 结果", "#F1F3F5", MID)
rounded_box(ax, (0.825, 0.13), 0.13, 0.15, "设备级\n结果", PALE_ORANGE, ORANGE)
arrow(ax, (0.225, 0.205), (0.315, 0.205), TEAL)
arrow(ax, (0.485, 0.205), (0.575, 0.205), TEAL)
arrow(ax, (0.765, 0.205), (0.825, 0.205), TEAL)
arrow(ax, (0.765, 0.60), (0.40, 0.28), NAVY)
ax.text(0.74, 0.46, "模型已训练", fontsize=7.4, color=NAVY, ha="center")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
save(fig, "fig1_method_pipeline")


pred = pd.read_csv(HERE / "uci361_recovered_predictions_2026-10-20.csv")
pred = pred.loc[pred.split.eq("one")].copy()
assert len(pred) == pred.file.nunique() == 192
pred["rank"] = pred.file.str.extract(r"_F0([123])0_").astype(int)
assert pred.groupby(["rank", "truth"]).size().eq(16).all()

# Figure 2: compact heat map avoids an empty three-point line plot while
# retaining the exact exposure-file error counts and one shared scale.
methods = ["原始 CNN", "逐试次归一化 CNN", "幅值增强 CNN"]
cols = ["raw_cnn", "norm_cnn", "aug_cnn"]
error_counts = np.array([
    [int((part.truth != part[col]).sum()) for _, part in pred.groupby("rank")]
    for col in cols
])
assert error_counts.tolist() == [[38, 18, 4], [1, 4, 3], [4, 1, 0]]
fig, ax = plt.subplots(figsize=(4.8, 2.35))
cmap = colors.LinearSegmentedColormap.from_list(
    "paper_blue", ["#F7FAFC", "#C8DBE6", "#77A9C1", NAVY]
)
im = ax.imshow(error_counts, cmap=cmap, norm=colors.Normalize(vmin=0, vmax=38),
               interpolation="none", aspect="auto")
ax.set_xticks(range(3), ["第1档", "第2档", "第3档"])
ax.set_yticks(range(3), methods)
ax.set_xlabel("目标设备低浓度第1—3档（每档64次）")
for (i, j), value in np.ndenumerate(error_counts):
    ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
            color="white" if value >= 14 else INK,
            fontweight="bold" if value <= 4 else "normal")
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.3)
ax.tick_params(which="minor", bottom=False, left=False)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=(0, 10, 20, 30, 38))
cbar.set_label("误判试次/次", fontsize=8)
fig.subplots_adjust(left=0.29, right=0.90, bottom=0.22, top=0.96)
save(fig, "fig2_low_rank_error_heatmap")


# Figure 3: shared-scale confusion matrices from complete held-device trials.
codes = ["CO", "Me", "Ea", "Ey"]
names = ["CO", "甲烷", "乙醇", "乙烯"]
matrices = [confusion_matrix(pred.truth, pred[column], labels=codes)
            for column in ("raw_cnn", "aug_cnn")]
assert [int(np.trace(matrix)) for matrix in matrices] == [132, 187]
fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.85), sharex=True, sharey=True)
for ax, matrix, title in zip(axes, matrices, ("（a）原始 CNN", "（b）幅值增强 CNN")):
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=48, interpolation="none", aspect="equal")
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    ax.set_xlabel("预测类别")
    ax.set_title(title, pad=6)
    for (i, j), value in np.ndenumerate(matrix):
        if value:
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.5,
                    color="white" if value >= 25 else INK)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
axes[0].set_ylabel("真实类别")
fig.subplots_adjust(left=0.095, right=0.89, bottom=0.16, top=0.88, wspace=0.12)
cbar = fig.colorbar(im, ax=axes, fraction=0.042, pad=0.022, ticks=(0, 12, 24, 36, 48))
cbar.set_label("供气试次/次", fontsize=8)
save(fig, "fig3_confusion_matrices")


# Figure 4: equal visual weight per physical device. Values are accuracies,
# not pooled trial counts, and the number of test trials remains explicit.
audit = json.loads((HERE / "uci361_device_cluster_audit_2026_10_23.json").read_text())
devices = list(range(1, 6))
raw = np.array([audit["per_device"][str(i)]["raw_accuracy"] for i in devices])
aug = np.array([audit["per_device"][str(i)]["augmented_accuracy"] for i in devices])
n_test = [audit["per_device"][str(i)]["exposure_files"] for i in devices]
assert np.all(aug > raw)
y = np.arange(len(devices))
fig, ax = plt.subplots(figsize=(5.15, 2.75))
for yi, left, right in zip(y, raw, aug):
    ax.plot([left, right], [yi, yi], color="#AEB8BF", lw=1.8, zorder=1)
ax.scatter(raw, y, s=36, marker="o", color=MID, edgecolor="white", linewidth=0.6,
           zorder=3, label="原始 SVM")
ax.scatter(aug, y, s=42, marker="D", color=ORANGE, edgecolor="white", linewidth=0.6,
           zorder=3, label="幅值增强 SVM")
for yi, left, right in zip(y, raw, aug):
    ax.text(right + 0.012, yi, f"+{(right-left)*100:.1f} pp", va="center", ha="left",
            fontsize=7.5, color=ORANGE)
ax.set_yticks(y, [f"设备{i}（{n}次）" for i, n in zip(devices, n_test)])
ax.invert_yaxis()
ax.set_xlim(0.55, 1.075)
ticks = np.arange(0.6, 1.01, 0.1)
ax.set_xticks(ticks, [f"{value:.1f}" for value in ticks])
ax.set_xlabel("正确试次/测试试次")
ax.grid(axis="x", color=GRID, linewidth=0.65, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.52, 1.00), ncol=2,
          frameon=False, fontsize=8, columnspacing=1.5, handletextpad=0.5)
fig.subplots_adjust(left=0.22, right=0.93, bottom=0.20, top=0.84)
save(fig, "fig4_device_paired_accuracy")

print(f"Wrote 8 publication files to {OUT}")
