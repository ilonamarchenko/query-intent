"""
Styled validation table as a figure: library check on 8 example queries.

Run from project root:
    python src/models/plot_validation.py
"""
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
PLOTS_DIR = Path("results/plots")

ROWS = [
    ("Що таке рекурсія і навіщо вона потрібна?", "навчання", "навчання", 0.98),
    ("Чим відрізняється list від tuple в Python?", "навчання", "навчання", 0.99),
    ("Напиши функцію бінарного пошуку на Python", "рішення", "рішення", 0.96),
    ("Реалізуй стек за допомогою масиву", "рішення", "рішення", 0.97),
    ("Як додати елемент до списку, якщо його там немає?", "рішення", "навчання", 0.98),
    ("Як працює алгоритм Дейкстри?", "навчання", "навчання", 0.97),
    ("Що таке замикання у JavaScript?", "навчання", "навчання", 0.95),
    ("Як відсортувати список словників за ключем?", "рішення", "навчання", 0.99),
]

CORRECT_BG = "#E3F0E6"
WRONG_BG   = "#F7E2DD"
HEADER_BG  = "#D7E3EF"
GREEN_TXT  = "#2E7D5B"
RED_TXT    = "#C0584B"

headers = ["Запит", "Очікуваний", "Передбачений", "Впевненість", ""]
col_w   = [0.50, 0.155, 0.155, 0.12, 0.07]
x = [sum(col_w[:i]) for i in range(len(col_w))]

n = len(ROWS)
row_h = 0.95
fig_h = (n + 1.4) * 0.62
fig, ax = plt.subplots(figsize=(11.5, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, n + 1)
ax.axis("off")

y_head = n
for j, h in enumerate(headers):
    ax.add_patch(plt.Rectangle((x[j], y_head), col_w[j], row_h,
                               facecolor=HEADER_BG, edgecolor="white", linewidth=1.5))
    if h:
        ax.text(x[j] + col_w[j] / 2, y_head + row_h / 2, h,
                ha="center", va="center", fontsize=11, fontweight="bold", color="#34495E")

for i, (q, exp, pred, conf) in enumerate(ROWS):
    y = n - 1 - i
    correct = exp == pred
    bg = CORRECT_BG if correct else WRONG_BG
    for j in range(len(headers)):
        ax.add_patch(plt.Rectangle((x[j], y), col_w[j], row_h,
                                   facecolor=bg, edgecolor="white", linewidth=1.5))

    ax.text(x[0] + 0.012, y + row_h / 2, q, ha="left", va="center",
            fontsize=10, color="#2B2B2B")

    ax.text(x[1] + col_w[1] / 2, y + row_h / 2, exp, ha="center", va="center",
            fontsize=10, color="#2B2B2B")
    pred_color = GREEN_TXT if correct else RED_TXT
    ax.text(x[2] + col_w[2] / 2, y + row_h / 2, pred, ha="center", va="center",
            fontsize=10, color=pred_color, fontweight="bold" if not correct else "normal")

    ax.text(x[3] + col_w[3] / 2, y + row_h / 2, f"{conf:.2f}", ha="center", va="center",
            fontsize=10, color="#2B2B2B")

    mark = "✓" if correct else "✗"
    ax.text(x[4] + col_w[4] / 2, y + row_h / 2, mark, ha="center", va="center",
            fontsize=15, color=GREEN_TXT if correct else RED_TXT, fontweight="bold")

fig.tight_layout()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
out = PLOTS_DIR / "09_validation.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
