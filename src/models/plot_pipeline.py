"""
Research-stages flowchart (pastel) for the introduction.

Run from project root:
    python src/models/plot_pipeline.py
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})

PLOTS_DIR = Path("results/plots")

STAGES = [
    ("1. Огляд літератури та\nпостановка задачі",
     "аналіз підходів до класифікації намірів"),
    ("2. Формування та розмітка\nдатасету (UK + EN)",
     "збір з форумів, синтетики, ручна розмітка 0/1"),
    ("3. Донавчання\nтрансформерних моделей",
     "mBERT, XLM-R, mDeBERTa, ukr-RoBERTa"),
    ("4. Порівняльний аналіз та\nпошук гіперпараметрів",
     "109 запусків, криві навчання"),
    ("5. Аналіз помилок і\nкрос-лінгвальне оцінювання",
     "матриця помилок, перенесення між мовами"),
    ("6. Розробка програмної\nбібліотеки query-intent",
     "інтеграція у навчальні системи"),
]

BOX_COLORS = ["#CFE3F0", "#D6ECD8", "#FBEBCF", "#F0DCE8", "#E2DCEF", "#D2ECE6"]
EDGE_COLORS = ["#7FA8D0", "#84C7A8", "#E8C06A", "#D499BE", "#B095D0", "#7FC4B8"]

fig, ax = plt.subplots(figsize=(7.8, 9.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(STAGES) * 2.2 + 0.5)
ax.axis("off")

n = len(STAGES)
box_w, box_h = 7.6, 1.55
x0 = (10 - box_w) / 2

for i, (title, sub) in enumerate(STAGES):
    y = (n - 1 - i) * 2.2 + 0.6
    box = FancyBboxPatch(
        (x0, y), box_w, box_h,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor=BOX_COLORS[i], edgecolor=EDGE_COLORS[i], linewidth=2,
    )
    ax.add_patch(box)
    ax.text(x0 + box_w / 2, y + box_h * 0.62, title,
            ha="center", va="center", fontsize=12, fontweight="bold", color="#2B2B2B")
    ax.text(x0 + box_w / 2, y + box_h * 0.20, sub,
            ha="center", va="center", fontsize=9.5, color="#555", style="italic")

    if i < n - 1:
        y_next_top = (n - 1 - (i + 1)) * 2.2 + 0.6 + box_h
        arrow = FancyArrowPatch(
            (5, y), (5, y_next_top),
            arrowstyle="-|>", mutation_scale=22,
            color="#9AA7B0", linewidth=2,
        )
        ax.add_patch(arrow)

fig.tight_layout()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
out = PLOTS_DIR / "00_research_stages.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
