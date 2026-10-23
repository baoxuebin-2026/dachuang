"""Generate restrained, single-figure charts for the Chinese journal draft.

The figures use only frozen experiment outputs.  Layout, labels, and colours
are deliberately conventional: one chart per file, thin axes, small type, no
decorative frame, and no synthetic uncertainty bars.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "论文结果图_期刊重构版_2026-10-24"
OUT.mkdir(exist_ok=True)

FONT = HERE / "assets/NotoSansSC-figure-subset.otf"
font_manager.fontManager.addfont(str(FONT))
FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()
MM = 1 / 25.4

BLUE = "#4C78A8"
ORANGE = "#E17C05"
GRAY = "#A7B0BA"
INK = "#333333"
LIGHT = "#E6E9ED"

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def style_axis(ax, *, horizontal_grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(direction="out", colors=INK, pad=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.set_axisbelow(True)
    if horizontal_grid:
        ax.grid(axis="y", color=LIGHT, linewidth=0.55)


def save(fig, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            OUT / f"{name}.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.035,
        )
    plt.close(fig)


def add_values(ax, bars, *, fmt="{:.3f}", dy=0.012) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )


preflight = json.loads((HERE / "paper_preflight_2026_10_22.json").read_text())
methods = preflight["methods"]
normalization = json.loads(
    (HERE / "paper_normalization_control_2026_10_22.json").read_text()
)
device_audit = json.loads(
    (HERE / "uci361_device_cluster_audit_2026_10_23.json").read_text()
)
pred = pd.read_csv(HERE / "uci361_recovered_predictions_2026-10-20.csv")
pred = pred.loc[pred.split.eq("one")].copy()
pred["rank"] = pred.file.str.extract(r"_F0([123])0_").astype(int)
assert len(pred) == pred.file.nunique() == 192

# Figure 1: main model/preprocessing comparison.
groups = ["原始", "RMS归一化", "幅值增强"]
svm = np.array([
    methods["svm_no_augmentation"]["macro_f1"],
    normalization["methods"]["normalized"]["macro_f1"],
    methods["svm_three_scale"]["macro_f1"],
])
cnn = np.array([
    methods["cnn_no_augmentation"]["macro_f1"],
    methods["cnn_per_record_rms"]["macro_f1"],
    methods["cnn_three_scale"]["macro_f1"],
])
x = np.arange(len(groups))
w = 0.31
fig, ax = plt.subplots(figsize=(118 * MM, 70 * MM))
b1 = ax.bar(x - w / 2, svm, w, label="SVM", color=BLUE)
b2 = ax.bar(x + w / 2, cnn, w, label="CNN", color=ORANGE)
ax.set_ylabel("宏平均 F1")
ax.set_xticks(x, groups)
ax.set_ylim(0, 1.06)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.25,
          columnspacing=1.4)
add_values(ax, b1)
add_values(ax, b2)
style_axis(ax)
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.19, top=0.94)
save(fig, "fig1_model_comparison")

# Figure 2: full-file errors at each low-concentration level.
series = []
for column in ("raw_cnn", "norm_cnn", "aug_cnn"):
    series.append(np.array([
        int((part.truth != part[column]).sum())
        for _, part in pred.groupby("rank", sort=True)
    ]))
labels = ["原始", "RMS归一化", "幅值增强"]
colors = [GRAY, BLUE, ORANGE]
x = np.arange(3)
w = 0.23
fig, ax = plt.subplots(figsize=(118 * MM, 70 * MM))
for idx, (values, label, color) in enumerate(zip(series, labels, colors)):
    bars = ax.bar(x + (idx - 1) * w, values, w, label=label, color=color)
    add_values(ax, bars, fmt="{:.0f}", dy=0.65)
ax.set_ylabel("误判文件数")
ax.set_xlabel("低浓度档")
ax.set_xticks(x, ["第1档", "第2档", "第3档"])
ax.set_ylim(0, 42)
ax.set_yticks([0, 10, 20, 30, 40])
ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.25)
style_axis(ax)
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.21, top=0.94)
save(fig, "fig2_errors_by_level")

# Figure 3: final CNN confusion matrix; one matrix keeps the page uncluttered.
class_codes = ["CO", "Me", "Ea", "Ey"]
class_names = ["CO", "甲烷", "乙醇", "乙烯"]
matrix = confusion_matrix(pred.truth, pred.aug_cnn, labels=class_codes)
assert matrix.sum() == 192 and list(matrix.diagonal()) == [48, 48, 47, 44]
fig, ax = plt.subplots(figsize=(82 * MM, 73 * MM))
image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=48, interpolation="nearest")
for row in range(4):
    for col in range(4):
        value = matrix[row, col]
        ax.text(col, row, str(value), ha="center", va="center", fontsize=8,
                color="white" if value >= 25 else INK)
ax.set_xticks(np.arange(4), class_names)
ax.set_yticks(np.arange(4), class_names)
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
ax.tick_params(length=0, pad=4)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0, 12, 24, 36, 48])
cbar.outline.set_linewidth(0.7)
cbar.ax.tick_params(direction="out", length=2.5, width=0.7, labelsize=7)
fig.subplots_adjust(left=0.20, right=0.90, bottom=0.16, top=0.985)
save(fig, "fig3_augmented_cnn_confusion")

# Figure 4: device-wise paired result, rendered as conventional grouped bars.
devices = np.arange(1, 6)
raw = np.array([device_audit["per_device"][str(i)]["raw_accuracy"] for i in devices])
aug = np.array([device_audit["per_device"][str(i)]["augmented_accuracy"] for i in devices])
counts = [device_audit["per_device"][str(i)]["exposure_files"] for i in devices]
assert counts == [48, 48, 48, 24, 24]
x = np.arange(5)
w = 0.33
fig, ax = plt.subplots(figsize=(118 * MM, 70 * MM))
b1 = ax.bar(x - w / 2, raw, w, label="原始", color=GRAY)
b2 = ax.bar(x + w / 2, aug, w, label="幅值增强", color=ORANGE)
ax.set_ylabel("正确数/测试数")
ax.set_xlabel("留出设备")
ax.set_xticks(x, [f"设备{i}\n(n={n})" for i, n in zip(devices, counts)])
ax.set_ylim(0, 1.08)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
          ncol=2, handlelength=1.25, columnspacing=1.4)
add_values(ax, b1)
add_values(ax, b2)
style_axis(ax)
fig.subplots_adjust(left=0.12, right=0.985, bottom=0.25, top=0.86)
save(fig, "fig4_device_accuracy")

print(f"Exported journal figures to {OUT}")
