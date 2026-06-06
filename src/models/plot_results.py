"""
Visualize experiment results from results/summary.csv.

Run from project root:
    python src/models/plot_results.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

MODEL_SHORT = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base":             "XLM-R-base",
    "microsoft/mdeberta-v3-base":   "mDeBERTa",
    "xlm-roberta-large":            "XLM-R-large",
    "ukr-detect/ukr-roberta-base":  "ukr-RoBERTa",
}
DATASET_LABEL = {"uk": "Українська", "en": "Англійська", "both": "UK + EN"}

COLORS = ["#7FA8D0", "#E8A598", "#84C7A8", "#BBA6DC", "#F0C58A"]

def load() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df["model_short"] = df["model"].map(lambda m: MODEL_SHORT.get(m, m.split("/")[-1]))
    return df

def plot_model_comparison(df: pd.DataFrame):
    """Main thesis table: test F1 for each model × dataset."""
    main = df[df["train_size"].astype(str) == "full"].copy()
    datasets = [d for d in ["uk", "en", "both"] if d in main["dataset"].unique()]
    models = list(dict.fromkeys(main["model_short"]))

    x = np.arange(len(datasets))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model in enumerate(models):
        vals = []
        for ds in datasets:
            row = main[(main["model_short"] == model) & (main["dataset"] == ds)]
            vals.append(row["test_f1"].values[0] if len(row) else np.nan)
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model,
                      color=COLORS[i % len(COLORS)], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABEL.get(d, d) for d in datasets], fontsize=12)
    ax.set_ylabel("F1 (macro)", fontsize=12)
    ax.set_title("Порівняння моделей за датасетами", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "01_model_comparison.png")

def plot_learning_curves(df: pd.DataFrame):
    """F1 vs training size for each model on each dataset."""
    size_map = {"100": 100, "200": 200, "300": 300, "full": 849}
    sub = df[df["max_length"].astype(str) == str(df["max_length"].mode()[0])].copy()
    sub = sub[sub["lr"].round(7) == sub["lr"].mode()[0]]
    sub["size_num"] = sub["train_size"].astype(str).map(size_map)
    sub = sub.dropna(subset=["size_num"])

    datasets = [d for d in ["uk", "en", "both"] if d in sub["dataset"].unique()]
    models = list(dict.fromkeys(sub["model_short"]))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for i, model in enumerate(models):
            m = sub[(sub["model_short"] == model) & (sub["dataset"] == ds)]
            m = m.sort_values("size_num")
            if m.empty:
                continue
            ax.plot(m["size_num"], m["test_f1"], marker="o", linewidth=2.5,
                    markersize=8, color=COLORS[i % len(COLORS)], label=model,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.set_xlabel("Розмір навчальної вибірки", fontsize=11)
        ax.set_ylabel("F1 (macro)" if ax is axes[0] else "", fontsize=11)
        ax.set_title(DATASET_LABEL.get(ds, ds), fontsize=12, fontweight="bold")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Криві навчання: вплив обсягу навчальних даних", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_learning_curves.png")

def plot_confusion_matrices(df: pd.DataFrame):
    """Confusion matrices loaded from saved test_metrics.json."""
    main = df[df["train_size"].astype(str) == "full"]
    datasets = [d for d in ["uk", "en", "both"] if d in main["dataset"].unique()]

    best_runs = {}
    for ds in datasets:
        sub = main[main["dataset"] == ds]
        if sub.empty:
            continue
        best_row = sub.loc[sub["test_f1"].idxmax()]
        cm_path = Path(best_row["run_dir"]) / "test_metrics.json"
        if cm_path.exists():
            with open(cm_path) as f:
                best_runs[ds] = (best_row["model_short"], json.load(f))

    if not best_runs:
        print("  No confusion matrices found yet — skipping plot 3")
        return

    fig, axes = plt.subplots(1, len(best_runs), figsize=(4.5 * len(best_runs), 4))
    if len(best_runs) == 1:
        axes = [axes]

    labels = ["Solution", "Learning"]
    for ax, (ds, (model, metrics)) in zip(axes, best_runs.items()):
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10, rotation=45)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title(f"{DATASET_LABEL.get(ds, ds)}\n({model})", fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                        ha="center", va="center", fontsize=10, color=color)
        plt.colorbar(im, ax=ax)

    fig.suptitle("Confusion Matrices — Best Model per Dataset",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_confusion_matrices.png")

def plot_metric_breakdown(df: pd.DataFrame):
    """Precision, Recall, F1 for best config of each model on UK dataset."""
    main = df[(df["train_size"].astype(str) == "full") & (df["dataset"] == "uk")]
    if main.empty:
        print("  No UK full-dataset results yet — skipping plot 4")
        return

    models = list(dict.fromkeys(main["model_short"]))
    metrics = ["test_precision", "test_recall", "test_f1"]
    labels  = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.7 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        row = main[main["model_short"] == model]
        if row.empty:
            continue
        row = row.loc[row["test_f1"].idxmax()]
        vals = [row[m] for m in metrics]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model,
                      color=COLORS[i % len(COLORS)], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score (macro)", fontsize=12)
    ax.set_title("Precision / Recall / F1 on Ukrainian Dataset", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    _save(fig, "04_metric_breakdown_uk.png")

def plot_crosslingual():
    """In-domain vs cross-lingual F1, grouped bars."""
    configs = [
        ("UK→UK", 0.938, "in"),
        ("UK→EN", 0.795, "cross"),
        ("EN→EN", 0.982, "in"),
        ("EN→UK", 0.794, "cross"),
    ]
    labels = [c[0] for c in configs]
    vals   = [c[1] for c in configs]

    bar_colors = ["#84C7A8" if c[2] == "in" else "#E8A598" for c in configs]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white",
                  linewidth=1.2, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                color="#333", fontweight="medium")

    ax.set_ylabel("F1 (macro)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Крос-лінгвальне перенесення (mBERT)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#84C7A8", label="У межах мови (in-domain)"),
        Patch(facecolor="#E8A598", label="Крос-лінгвально (cross-lingual)"),
    ], fontsize=10, loc="lower center", framealpha=0.9)

    fig.tight_layout()
    _save(fig, "07_crosslingual.png")

def plot_error_types():
    """FN vs FP confusion as a pastel confusion matrix for the UK best model."""

    cm = np.array([[99, 5], [6, 72]])
    labels = ["Рішення", "Навчання"]

    from matplotlib.colors import LinearSegmentedColormap
    pastel = LinearSegmentedColormap.from_list("p", ["#FBF3E4", "#84C7A8"])

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap=pastel, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=20, color="#2B2B2B", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=11, rotation=90, va="center")
    ax.set_xlabel("Передбачений клас", fontsize=12)
    ax.set_ylabel("Справжній клас", fontsize=12)
    ax.set_title("Матриця помилок (mBERT, українська)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="both", length=0)
    fig.tight_layout()
    _save(fig, "08_confusion_uk.png")

def _save(fig, name: str):
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"Loaded {len(df)} runs from {SUMMARY_CSV}")
    print("Generating plots...")
    plot_model_comparison(df)
    plot_learning_curves(df)
    plot_confusion_matrices(df)
    plot_metric_breakdown(df)
    plot_crosslingual()
    plot_error_types()
    print(f"\nDone → {PLOTS_DIR}/")
