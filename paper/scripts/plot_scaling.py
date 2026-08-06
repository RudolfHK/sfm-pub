"""Erzeugt die Skalierungsabbildung für paper_t.md.

Alle Werte stammen aus EVALUATION_RESULTS.md (Abschnitte A.1 und A.2) und sind
hier fest hinterlegt, damit die Abbildung ohne erneuten Messlauf reproduzierbar
ist.

    python paper/scripts/plot_scaling.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures" / "abb11_skalierung.png"

# EVALUATION_RESULTS.md, A.2
N_IMAGES = np.array([6, 20, 67])
PAIRS = np.array([15, 190, 2211])
T_MATCHING = np.array([7.6, 94.4, 1036.6])
T_TOTAL = np.array([13.1, 113.0, 1135.5])

# EVALUATION_RESULTS.md, A.1 (Zeilen n06_base, n20_base, n67_base)
STAGES = {
    "Merkmale": np.array([3.8, 13.4, 41.8]),
    "Matching": np.array([7.6, 94.4, 1036.6]),
    "Verifikation": np.array([0.0, 1.0, 8.7]),
    "Rekonstruktion + BA": np.array([0.0, 1.2, 38.2]),
    "Export": np.array([0.2, 1.6, 8.0]),
}
COLORS = ["#8ecae6", "#1f6f8b", "#f4a261", "#e76f51", "#adb5bd"]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- links: Laufzeit ueber Bildzahl, mit O(N^2)-Referenz -----------------
    ax1.plot(N_IMAGES, T_TOTAL, "o-", color="#1f6f8b", lw=2, label="Gesamtlaufzeit")
    ax1.plot(N_IMAGES, T_MATCHING, "s--", color="#e76f51", lw=2, label="davon Matching")

    n_ref = np.linspace(5, 70, 200)
    scale = T_MATCHING[-1] / (N_IMAGES[-1] ** 2)
    ax1.plot(n_ref, scale * n_ref**2, ":", color="#6c757d", lw=1.4,
             label=r"Referenz $\propto N^2$")

    for n, t in zip(N_IMAGES, T_TOTAL):
        dx = -34 if n == N_IMAGES[-1] else 6
        ax1.annotate(f"{t:.0f} s", (n, t), textcoords="offset points",
                     xytext=(dx, 8), fontsize=9, color="#1f6f8b")

    ax1.set_xlabel("Anzahl Eingabebilder N")
    ax1.set_xlim(2, 74)
    ax1.set_ylabel("Laufzeit (s)")
    ax1.set_title("Laufzeit wächst quadratisch mit der Bildzahl", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9, frameon=False)

    # --- rechts: Anteil der Stufen an der Gesamtlaufzeit ---------------------
    totals = sum(STAGES.values())
    bottom = np.zeros(3)
    x = np.arange(3)
    for (name, vals), color in zip(STAGES.items(), COLORS):
        share = 100 * vals / totals
        ax2.bar(x, share, 0.55, bottom=bottom, label=name, color=color)
        for xi, (s, b) in enumerate(zip(share, bottom)):
            if s >= 6:
                ax2.text(xi, b + s / 2, f"{s:.0f} %", ha="center", va="center",
                         fontsize=9, color="white", fontweight="bold")
        bottom += share

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{n} Bilder" for n in N_IMAGES])
    ax2.set_ylabel("Anteil an der Laufzeit (%)")
    ax2.set_ylim(0, 118)
    ax2.set_title("Matching verdrängt alle anderen Stufen", fontsize=11)
    ax2.legend(fontsize=8, frameon=False, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0))
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"geschrieben: {OUT}")


if __name__ == "__main__":
    main()
