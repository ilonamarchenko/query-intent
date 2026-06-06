"""
Pastel heatmap matrices: model x dataset x metric.

Run from project root:
    python src/models/plot_heatmap.py
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

MODEL_SHORT = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base":             "XLM-R-base",
    "microsoft/mdeberta-v3-base":   "mDeBERTa",
    "xlm-roberta-large":            "XLM-R-large",
    "youscan/ukr-roberta-base":     "ukr-RoBERTa",
}
MODEL_ORDER = ["mBERT", "XLM-R-base", "mDeBERTa", "XLM-R-large", "ukr-RoBERTa"]
DATASETS = [("uk", "Українська"), ("en", "Англійська"), ("both", "UK + EN")]
METRICS = [("test_precision", "P"), ("test_recall", "R"), ("test_f1", "F1")]

PASTEL = LinearSegmentedColormap.from_list(
    "pastel_green",
    ["#FBF3E4", "#F3E7C9", "#CFE6C3", "#A8D8B9", "#84C7A8"],
)

def load_main() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[(df["train_size"].astype(str) == "full")
            & (df["max_length"] == 256)
            & (df["lr"].round(7) == 2e-5)].copy()
    df["model_short"] = df["model"].map(lambda m: MODEL_SHORT.get(m, m.split("/")[-1]))
    return df

def plot_full_matrix(df):
    cols, col_labels, col_groups = [], [], []
    for ds, ds_lbl in DATASETS:
        for mk, m_lbl in METRICS:
            cols.append((ds, mk))
            col_labels.append(m_lbl)
            col_groups.append(ds_lbl)

    data = np.full((len(MODEL_ORDER), len(cols)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, (ds, mk) in enumerate(cols):
            row = df[(df["model_short"] == model) & (df["dataset"] == ds)]
            if len(row):
                data[i, j] = row[mk].values[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    vmin, vmax = 0.85, 1.0
    im = ax.imshow(data, cmap=PASTEL, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(len(MODEL_ORDER)):
        for j in range(len(cols)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=10, color="#2B2B2B", fontweight="medium")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="#BBB")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=11)

    for k, (ds, ds_lbl) in enumerate(DATASETS):
        center = k * len(METRICS) + (len(METRICS) - 1) / 2
        ax.text(center, -0.75, ds_lbl, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#444")
        if k > 0:
            ax.axvline(k * len(METRICS) - 0.5, color="white", linewidth=3)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Значення метрики", fontsize=10)

    ax.set_title("Якість моделей за датасетами та метриками",
                 fontsize=13, fontweight="bold", pad=28)
    fig.tight_layout()
    _save(fig, "05_metrics_matrix.png")

def plot_f1_matrix(df):
    data = np.full((len(MODEL_ORDER), len(DATASETS)), np.nan)
    for i, model in enumerate(MODEL_ORDER):
        for j, (ds, _) in enumerate(DATASETS):
            row = df[(df["model_short"] == model) & (df["dataset"] == ds)]
            if len(row):
                data[i, j] = row["test_f1"].values[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=PASTEL, vmin=0.85, vmax=1.0, aspect="auto")

    for i in range(len(MODEL_ORDER)):
        for j in range(len(DATASETS)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=13, color="#2B2B2B", fontweight="medium")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=13, color="#BBB")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([d[1] for d in DATASETS], fontsize=12)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=12)

    ax.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("F1 (macro)", fontsize=11)

    ax.set_title("Макро F1 за моделями та датасетами", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_f1_matrix.png")

def _save(fig, name):
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_main()
    print(f"Loaded {len(df)} main runs")
    plot_full_matrix(df)
    plot_f1_matrix(df)
    print(f"Done → {PLOTS_DIR}/")
