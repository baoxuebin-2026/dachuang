"""Rebuild the four manuscript figures with SciPlot Academic conventions.

Only frozen outputs are plotted. The script uses the Nature preset for axis
geometry, inward ticks and export defaults, then applies the bundled Chinese
font and a restrained, print-safe palette. No observations are synthesized.
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
import sciplot as sp
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "论文结果图_SciPlot版_2026-10-23"
OUT.mkdir(exist_ok=True)
FONT = HERE / "assets/NotoSansSC-figure-subset.otf"
font_manager.fontManager.addfont(str(FONT))
FONT_NAME = font_manager.FontProperties(fname=str(FONT)).get_name()
MM = 1 / 25.4

GRAY = "#A7B4C2"
BLUE = "#5E98C2"
ORANGE = "#D86B35"
BLACK = "#202020"

sp.setup_style("nature", palette="earth", lang="zh")
plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "svg.fonttype": "path",
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def finish(fig, name: str) -> None:
    """Export one compact single-column figure and run SciPlot's audit."""
    sp.save(
        fig,
        name,
        dir=str(OUT),
        formats=("png", "svg", "pdf"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        close=True,
        audit=True,
    )


def polish(ax) -> None:
    """Keep the complete Nature frame and remove decorative grid lines."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.72)
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, width=0.72)


preflight = json.loads((HERE / "paper_preflight_2026_10_22.json").read_text())
methods = preflight["methods"]
pred = pd.read_csv(HERE / "uci361_recovered_predictions_2026-10-20.csv")
pred = pred.loc[pred.split.eq("one")].copy()
assert len(pred) == pred.file.nunique() == 192
pred["rank"] = pred.file.str.extract(r"_F0([123])0_").astype(int)
assert pred.groupby(["rank", "truth"]).size().eq(16).all()
audit = json.loads((HERE / "uci361_device_cluster_audit_2026_10_23.json").read_text())

# Fig. 1: classifier and preprocessing comparison.
norm = json.loads((HERE / "paper_normalization_control_2026_10_22.json").read_text())
svm = [methods["svm_no_augmentation"]["macro_f1"],
       norm["methods"]["normalized"]["macro_f1"],
       methods["svm_three_scale"]["macro_f1"]]
cnn = [methods["cnn_no_augmentation"]["macro_f1"],
       methods["cnn_per_record_rms"]["macro_f1"],
       methods["cnn_three_scale"]["macro_f1"]]
result = sp.plot_grouped_bar(
    groups=["原始", "RMS归一化", "幅值增强"],
    data={"SVM": svm, "CNN": cnn},
    ylabel="宏平均 F1",
    colors=[GRAY, ORANGE],
    width=0.70,
    show_values=True,
    value_fmt=".3f",
    legend_loc="upper left",
    venue="nature",
    lang="zh",
)
fig, ax = result
fig.set_size_inches(90 * MM, 62 * MM)
fig.subplots_adjust(left=.16, right=.98, bottom=.20, top=.91)
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, .2))
ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.5,
          columnspacing=1.2)
polish(ax)
finish(fig, "fig1_model_comparison")

# Fig. 2: errors by low-concentration rank.
error_data = {}
for column, label in (("raw_cnn", "原始"),
                      ("norm_cnn", "RMS归一化"),
                      ("aug_cnn", "幅值增强")):
    error_data[label] = [int((part.truth != part[column]).sum())
                         for _, part in pred.groupby("rank", sort=True)]
result = sp.plot_grouped_bar(
    groups=["第1档", "第2档", "第3档"],
    data=error_data,
    ylabel="误判次数",
    colors=[GRAY, BLUE, ORANGE],
    width=0.76,
    show_values=True,
    value_fmt=".0f",
    legend_loc="upper right",
    venue="nature",
    lang="zh",
)
fig, ax = result
fig.set_size_inches(90 * MM, 62 * MM)
fig.subplots_adjust(left=.16, right=.98, bottom=.20, top=.91)
ax.set_ylim(0, 45)
ax.set_yticks([0, 10, 20, 30, 40])
ax.legend(loc="upper right", ncol=1, frameon=False, handlelength=1.5)
polish(ax)
finish(fig, "fig2_errors_by_level")

# Fig. 3: confusion matrix for the final augmented CNN.
labels = ["CO", "Me", "Ea", "Ey"]
names = ["CO", "甲烷", "乙醇", "乙烯"]
matrix = confusion_matrix(pred.truth, pred.aug_cnn, labels=labels)
assert matrix.sum() == 192 and list(matrix.diagonal()) == [48, 48, 47, 44]
result = sp.plot_heatmap(
    matrix,
    row_labels=names,
    col_labels=names,
    xlabel="预测类别",
    ylabel="真实类别",
    cmap="Blues",
    show_values=True,
    fmt=".0f",
    vmin=0,
    vmax=48,
    aspect="equal",
    venue="nature",
    lang="zh",
)
fig, ax = result
fig.set_size_inches(82 * MM, 69 * MM)
fig.subplots_adjust(left=.22, right=.89, bottom=.20, top=.96)
ax.tick_params(axis="both", which="both", top=False, right=False,
               bottom=False, left=False, length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
if len(fig.axes) > 1:
    colorbar_axis = fig.axes[1]
    colorbar_axis.minorticks_off()
    colorbar_axis.tick_params(which="major", direction="out", length=2.5,
                              width=0.72)
finish(fig, "fig3_augmented_cnn_confusion")

# Fig. 4: paired held-device result as conventional grouped bars.
devices = np.arange(1, 6)
raw = [audit["per_device"][str(i)]["raw_accuracy"] for i in devices]
augmented = [audit["per_device"][str(i)]["augmented_accuracy"] for i in devices]
counts = [audit["per_device"][str(i)]["exposure_files"] for i in devices]
assert counts == [48, 48, 48, 24, 24] and np.all(np.asarray(augmented) > raw)
result = sp.plot_grouped_bar(
    groups=[f"设备{i}" for i in devices],
    data={"原始": raw, "幅值增强": augmented},
    ylabel="正确数/测试数",
    colors=[GRAY, ORANGE],
    width=0.72,
    show_values=True,
    value_fmt=".3f",
    legend_loc="upper left",
    venue="nature",
    lang="zh",
)
fig, ax = result
fig.set_size_inches(90 * MM, 64 * MM)
fig.subplots_adjust(left=.16, right=.98, bottom=.20, top=.91)
ax.set_ylim(0, 1.14)
ax.set_yticks(np.arange(0, 1.01, .2))
ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.5,
          columnspacing=1.2)
polish(ax)
finish(fig, "fig4_device_accuracy")

print(f"Exported four SciPlot PNG/SVG/PDF figure sets to {OUT}")
