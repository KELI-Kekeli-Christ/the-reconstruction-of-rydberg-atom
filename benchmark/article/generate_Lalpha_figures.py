#!/usr/bin/env python3
"""Generate publication figures comparing formulations across the 3 L-α experiments."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PASQAL = ROOT / "test_real" / "results" / "exp_L_alpha_full_results.csv"
CSV_EXTENDED = ROOT / "test_real" / "results" / "exp_L_alpha_full_results_n4_50.csv"
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

FORMS = ["Standard", "Simple", "Simple_square", "Taylor", "Taylor_square"]
FORM_SHORT = {
    "Standard": "Standard",
    "Simple": "Simple",
    "Simple_square": "Simple²",
    "Taylor": "Taylor",
    "Taylor_square": "Taylor²",
}
SOURCES = [
    ("exp3_pasqal", "Pasqal (αL, n=25)"),
    ("exp2_Q_exacte_synthetique", "$Q$ exacte synthétique"),
    ("exp1_Q_aleatoire", "$Q$ aléatoire"),
]
# Distinct, non-purple academic palette
COLORS = {
    "Standard": "#1b4f72",
    "Simple": "#117a65",
    "Simple_square": "#196f3d",
    "Taylor": "#b9770e",
    "Taylor_square": "#922b21",
}

# Cap huge errors for log plots (still visibly "bad")
ERR_CAP = 1e20


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load_rows():
    rows = []
    for path, allowed_sources in (
        (CSV_PASQAL, {"exp3_pasqal"}),
        (
            CSV_EXTENDED,
            {"exp2_Q_exacte_synthetique", "exp1_Q_aleatoire"},
        ),
    ):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("source") in allowed_sources:
                    rows.append(row)

    for src, title in SOURCES:
        src_rows = [r for r in rows if r["source"] == src]
        if not src_rows:
            raise RuntimeError(f"Aucune ligne {src} dans les CSV de L-α")
        alphas = sorted({float(r["alpha"]) for r in src_rows})
        ns = sorted({int(float(r["n"])) for r in src_rows})
        print(f"{title}: {len(src_rows)} runs, n={ns}, α={alphas}")

    return rows


def style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def bucket(rows, source, use_zmin, form, alpha=None):
    out = []
    for r in rows:
        if r["source"] != source:
            continue
        if (r["use_zmin"] == "true") != use_zmin:
            continue
        if r["formulation"] != form:
            continue
        if alpha is not None and abs(float(r["alpha"]) - alpha) > 1e-15:
            continue
        out.append(r)
    return out


def med_err(rows):
    vals = []
    for r in rows:
        e = fnum(r["err_rydberg"])
        if e is None or e <= 0:
            continue
        vals.append(min(e, ERR_CAP))
    return median(vals) if vals else None


def match_rate(rows):
    if not rows:
        return None
    return 100.0 * sum(1 for r in rows if r["dmin_matches_zmin"] == "true") / len(rows)


def fig_best_by_experiment(rows):
    """Median Eryd (log) per formulation, avec vs sans zmin, α=1e-4."""
    alpha = 1e-4
    fig, axes = plt.subplots(
        len(SOURCES), 1, figsize=(6.4, 3.6 * len(SOURCES)), sharey=False
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(FORMS))
    w = 0.38

    for idx, (ax, (src, title)) in enumerate(zip(axes, SOURCES)):
        avec, sans = [], []
        for form in FORMS:
            avec.append(med_err(bucket(rows, src, True, form, alpha)) or np.nan)
            sans.append(med_err(bucket(rows, src, False, form, alpha)) or np.nan)
        avec = np.array(avec, dtype=float)
        sans = np.array(sans, dtype=float)

        bars1 = ax.bar(
            x - w / 2,
            avec,
            w,
            label="avec $z_{\\min}$",
            color="#2c3e50",
            edgecolor="none",
        )
        bars2 = ax.bar(
            x + w / 2,
            sans,
            w,
            label="sans $z_{\\min}$",
            color="#85929e",
            edgecolor="none",
        )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([FORM_SHORT[f] for f in FORMS], rotation=30, ha="right")
        ax.set_title(title, loc="left")
        if np.any(np.isfinite(avec)):
            i = int(np.nanargmin(avec))
            bars1[i].set_color("#0e6655")
            bars1[i].set_edgecolor("#0b5345")
            bars1[i].set_linewidth(1.2)
        if np.any(np.isfinite(sans)):
            j = int(np.nanargmin(sans))
            bars2[j].set_color("#1a5276")
            bars2[j].set_edgecolor("#154360")
            bars2[j].set_linewidth(1.2)

        ax.set_ylabel(r"médiane $\mathcal{E}_{\mathrm{Ryd}}$  ($\alpha=10^{-4}$)")
        if idx == 0:
            ax.legend(loc="upper left", frameon=False)
    fig.suptitle(
        "Quelle formulation minimise l'erreur Rydberg ?  (meilleure barre soulignée)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGDIR / "lalpha_best_err_by_source.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lalpha_best_err_by_source.png")
    plt.close(fig)
    print("wrote", out)


def fig_match_by_experiment(rows):
    """Concordance dmin≈zmin (%) per formulation, α=1e-4."""
    alpha = 1e-4
    fig, axes = plt.subplots(
        len(SOURCES), 1, figsize=(6.4, 3.6 * len(SOURCES)), sharey=True
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(FORMS))
    w = 0.38

    for idx, (ax, (src, title)) in enumerate(zip(axes, SOURCES)):
        avec, sans = [], []
        for form in FORMS:
            avec.append(match_rate(bucket(rows, src, True, form, alpha)) or 0)
            sans.append(match_rate(bucket(rows, src, False, form, alpha)) or 0)
        ax.bar(x - w / 2, avec, w, label=r"avec $z_{\min}$", color="#1a5276")
        ax.bar(x + w / 2, sans, w, label=r"sans $z_{\min}$", color="#7fb3d5")
        ax.set_ylim(0, 105)
        ax.axhline(100, color="#566573", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([FORM_SHORT[f] for f in FORMS], rotation=30, ha="right")
        ax.set_title(title, loc="left")

        ax.set_ylabel(r"concordance $d_{\min}\!\approx\!z_{\min}^{\mathrm{opt}}$ (\%)")
        if idx == 0:
            ax.legend(loc="lower left", frameon=False)
    fig.suptitle(
        r"Émergence de $z_{\min}^{\mathrm{opt}}$  ($\alpha=10^{-4}$, tolérance 20\%)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGDIR / "lalpha_match_by_source.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lalpha_match_by_source.png")
    plt.close(fig)
    print("wrote", out)


def fig_err_vs_alpha(rows):
    """Median Eryd vs α (avec zmin)."""
    alphas = [1e-6, 1e-4, 1e-2, 1.0, 5.0, 10.0]
    fig, axes = plt.subplots(
        len(SOURCES), 1, figsize=(6.4, 3.6 * len(SOURCES)), sharey=False
    )
    axes = np.atleast_1d(axes)

    for idx, (ax, (src, title)) in enumerate(zip(axes, SOURCES)):
        for form in FORMS:
            ys = []
            for a in alphas:
                m = med_err(bucket(rows, src, True, form, a))
                ys.append(m if m is not None else np.nan)
            ax.plot(
                alphas,
                ys,
                marker="o",
                ms=4,
                lw=1.6,
                color=COLORS[form],
                label=FORM_SHORT[form],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title + r" (avec $z_{\min}$)", loc="left")

        ax.set_ylabel(r"médiane $\mathcal{E}_{\mathrm{Ryd}}$")
        if idx == 0:
            ax.legend(loc="best", frameon=False, ncol=1)
    fig.suptitle(
        r"Sensibilité à $\alpha$ — erreur Rydberg médiane (avec contrainte $z_{\min}$)",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGDIR / "lalpha_err_vs_alpha.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lalpha_err_vs_alpha.png")
    plt.close(fig)
    print("wrote", out)


def fig_winner_summary(rows):
    """Summary chart: best formulation name with/without zmin."""
    alpha = 1e-4
    fig, ax = plt.subplots(figsize=(7.2, 2.9))

    scenarios = []
    # (label, source, use_zmin)
    for src, stitle in SOURCES:
        for uz, zlab in [(True, "avec $z_{\\min}$"), (False, "sans $z_{\\min}$")]:
            scenarios.append((f"{stitle}\n{zlab}", src, uz))

    y = np.arange(len(scenarios))
    best_errs = []
    best_names = []
    bar_colors = []
    for _, src, uz in scenarios:
        scores = []
        for form in FORMS:
            m = med_err(bucket(rows, src, uz, form, alpha))
            if m is not None:
                scores.append((m, form))
        if scores:
            m, form = min(scores, key=lambda t: t[0])
            best_errs.append(m)
            best_names.append(FORM_SHORT[form])
            bar_colors.append(COLORS[form])
        else:
            best_errs.append(np.nan)
            best_names.append("—")
            bar_colors.append("#bdc3c7")

    ax.barh(y, best_errs, color=bar_colors, height=0.65, edgecolor="white", lw=0.5)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in scenarios])
    ax.set_xlabel(r"meilleure médiane $\mathcal{E}_{\mathrm{Ryd}}$  ($\alpha=10^{-4}$)")
    ax.set_title("Formulation gagnante sur les trois jeux de données")
    for yi, (err, name) in enumerate(zip(best_errs, best_names)):
        if math.isfinite(err):
            ax.text(
                err * 1.4,
                yi,
                name,
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
                color="#1c2833",
            )
    # Legend of colors
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(facecolor=COLORS[f], label=FORM_SHORT[f]) for f in FORMS],
        loc="lower right",
        frameon=False,
        title="formulation",
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.18)
    out = FIGDIR / "lalpha_winners.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lalpha_winners.png")
    plt.close(fig)
    print("wrote", out)


def fig_placeholder_n450():
    """Empty framed placeholder for future n=4..50 tables/figures."""
    fig, ax = plt.subplots(figsize=(7.5, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        mpl.patches.FancyBboxPatch(
            (0.05, 0.15),
            0.9,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#f4f6f7",
            edgecolor="#5d6d7e",
            linewidth=1.2,
            linestyle="--",
        )
    )
    ax.text(
        0.5,
        0.55,
        "Placeholder — résultats $n\\in\\{4,10,20,30,40,50\\}$\n"
        "(run Joyeux en cours : "
        "\\texttt{exp\\_L\\_alpha\\_full\\_results\\_n4\\_50.csv})",
        ha="center",
        va="center",
        fontsize=11,
        color="#2c3e50",
    )
    ax.text(
        0.5,
        0.28,
        "Les tableaux détaillés seront mis à jour à la fin du calcul.",
        ha="center",
        va="center",
        fontsize=9,
        color="#566573",
        style="italic",
    )
    # Don't save as PDF with latex escapes badly - write a note tex instead
    plt.close(fig)


def write_placeholder_tex():
    out = Path(__file__).resolve().parent / "tables" / "exp_lalpha_n450_placeholder.tex"
    out.write_text(
    r"""% Auto-generated — note d'extension L-α
\begin{table}[H]
\centering
\caption{Extension $n\in\{4,10,20,30,40,50\}$ pour les données synthétiques et aléatoires.
Les tableaux détaillés sont générés dans \texttt{exp\_lalpha\_synth.tex} et
	exttt{exp\_lalpha\_random.tex} à partir de
	exttt{exp\_L\_alpha\_full\_results\_n4\_50.csv}.}
\label{tab:lalpha-n450-placeholder}
\footnotesize
\begin{tabular}{@{}llcccc@{}}
  \toprule
  Source & Formulation &
  \multicolumn{2}{c}{avec $z_{\min}$} &
  \multicolumn{2}{c}{sans $z_{\min}$} \\
  \cmidrule(lr){3-4}\cmidrule(lr){5-6}
   &  & Concord. & $\Eryd$ & Concord. & $\Eryd$ \\
  \midrule
    \multicolumn{6}{c}{\emph{(voir tableaux~\ref{tab:lalpha-overview}--\ref{tab:lalpha-rand} pour le détail complet)}} \\
  \bottomrule
\end{tabular}
\end{table}
""",
        encoding="utf-8",
    )
    print("wrote", out)


def main():
    style()
    rows = load_rows()
    print(
        "loaded "
        f"{len(rows)} rows from {CSV_PASQAL.name} + {CSV_EXTENDED.name}"
    )
    fig_best_by_experiment(rows)
    fig_match_by_experiment(rows)
    fig_err_vs_alpha(rows)
    fig_winner_summary(rows)
    write_placeholder_tex()
    print("done →", FIGDIR)


if __name__ == "__main__":
    main()
