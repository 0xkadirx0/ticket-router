"""Draws docs/architecture.png (R6). Block names, precisions and files match the code."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .common import ROOT


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2b4c7e", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444"))


def main():
    fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
    box(ax, 0.1, 2.3, 1.5, 1.0, "User\n(browser)\nGradio UI\napp.py", "#fff4d6", "#b58900", bold=True)
    box(ax, 2.0, 2.3, 2.2, 1.0, "ELIZA rules\nsrc/textproc.py\n(small talk → reply)", "#f3e8ff", "#6b3fa0")
    box(ax, 2.0, 0.6, 2.2, 1.3, "Pre-processing (regex)\nnormalize · model-code\nextract · edit-distance\ncorrection (rule-based)", "#f3e8ff", "#6b3fa0")
    box(ax, 5.0, 4.0, 3.2, 0.9, "1  tf-idf + LogReg  (sklearn, fp64)", "#e8f5e9", "#2e7d32")
    box(ax, 5.0, 2.9, 3.2, 0.9, "2  word2vec avg → FFN  (PyTorch, fp32)", "#e8f5e9", "#2e7d32")
    box(ax, 5.0, 1.8, 3.2, 0.9, "3  BERT-base fine-tuned  (fp32)", "#e8f5e9", "#2e7d32")
    box(ax, 5.0, 0.5, 3.2, 1.0, "4  DistilBERT-KD  (INT8 dynamic quant.)\n★ drives the routing decision", "#ffe0e0", "#c62828", bold=True)
    box(ax, 9.0, 2.3, 1.7, 1.0, "Post-processing\nsoftmax conf. +\nthreshold\nsrc/teams.py", "#e3f2fd", "#1565c0")
    box(ax, 11.0, 3.2, 1.9, 0.9, "Team (domain →\nteam map) or\nHuman agent", "#fff4d6", "#b58900", bold=True)
    box(ax, 11.0, 1.7, 1.9, 1.1, "Side-by-side table:\nintent · conf. · ms\nper model", "#fff4d6", "#b58900")
    arrow(ax, 1.6, 2.8, 2.0, 2.8); arrow(ax, 1.0, 2.3, 2.0, 1.4); arrow(ax, 4.2, 1.3, 5.0, 4.4)
    for y in (3.35, 2.25, 1.0): arrow(ax, 4.2, 1.25, 5.0, y + 0.3 if y > 1 else 1.0)
    for y in (4.45, 3.35, 2.25, 1.0): arrow(ax, 8.2, y, 9.0, 2.8)
    arrow(ax, 10.7, 2.9, 11.0, 3.6); arrow(ax, 10.7, 2.6, 11.0, 2.2)
    ax.text(0.1, 5.3, "Support Ticket Router: data flow from user input to output (runs locally on CPU)",
            fontsize=11, weight="bold")
    ax.text(0.1, 0.1, "Offline: src/train_*.py + src/quantize.py + src/evaluate.py  →  models/ (HF Hub) and results/results.json",
            fontsize=8, color="#555")
    out = ROOT / "docs"; out.mkdir(exist_ok=True)
    fig.savefig(out / "architecture.png", dpi=170, bbox_inches="tight"); print("wrote docs/architecture.png")


if __name__ == "__main__":
    main()
