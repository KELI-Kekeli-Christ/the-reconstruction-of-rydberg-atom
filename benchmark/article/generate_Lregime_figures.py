#!/usr/bin/env python3
"""
Graphes « meilleure formulation » pour L fixe / adaptatif / optimal
sur les 3 expériences (Q aléatoire, Q synthétique, Pasqal).
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = Path(__file__).resolve().parent
FIGDIR = ARTICLE / "figures"
FIGDIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ARTICLE))
from generate_result_tables import (  # noqa: E402
    EXP1_SUMMARY,
    PASQAL_CSV,
    EXP2_LOG,
    parse_exp2_summary,
    parse_err,
    normalize_formulation,
    keep_formulation,
)

FORMS = ["MDS_Exact", "Standard", "Simple", "Simple_square", "Taylor", "Taylor_square"]
NLP_FORMS = [f for f in FORMS if f != "MDS_Exact"]
FORM_SHORT = {
    "MDS_Exact": "MDS",
    "Standard": "Standard",
    "Simple": "Simple",
    "Simple_square": "Simple²",
    "Taylor": "Taylor",
    "Taylor_square": "Taylor²",
}
L_ORDER = ["L_fixe", "L_adaptatif", "L_optimal"]
L_SHORT = {"L_fixe": "Fixe", "L_adaptatif": "Adaptatif", "L_optimal": "Optimal"}
COLORS = {
    "MDS_Exact": "#2c3e50",
    "Standard": "#1a5276",
    "Simple": "#117a65",
    "Simple_square": "#196f3d",
    "Taylor": "#b9770e",
    "Taylor_square": "#922b21",
}
ERR_CAP = 1e50


def style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
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


def safe(e):
    if e is None or not math.isfinite(e) or e <= 0:
        return None
    return min(e, ERR_CAP)


# ── Data loaders → dict[L_type][form] = median Eryd ───────────

def load_exp1():
    """Aggregate EXP1_SUMMARY over n : median of mean Eryd per (L, form)."""
    buckets = defaultdict(lambda: defaultdict(list))
    for (n, lt), forms in EXP1_SUMMARY.items():
        for form, (eryd, _t) in forms.items():
            if form not in FORMS:
                continue
            e = safe(eryd)
            if e is not None:
                buckets[lt][form].append(e)
    out = {}
    for lt in L_ORDER:
        out[lt] = {f: median(buckets[lt][f]) for f in FORMS if buckets[lt][f]}
    return out


def load_exp2():
    """From exp2_benchmark.log averages ; take best strategy per form."""
    if not EXP2_LOG.exists():
        raise FileNotFoundError(EXP2_LOG)
    rows = parse_exp2_summary(EXP2_LOG)
    # key (L, form) → list of eryd (over n, strategies)
    best_per_n = defaultdict(dict)  # (lt, form, n) -> min eryd
    for r in rows:
        form = r["formulation"]
        if form not in FORMS:
            continue
        e = safe(r["obj_rydberg"])
        if e is None:
            continue
        key = (r["L_type"], form, r["n"])
        if key not in best_per_n or e < best_per_n[key]:
            best_per_n[key] = e
    buckets = defaultdict(lambda: defaultdict(list))
    for (lt, form, n), e in best_per_n.items():
        buckets[lt][form].append(e)
    out = {}
    for lt in L_ORDER:
        out[lt] = {f: median(buckets[lt][f]) for f in FORMS if buckets[lt][f]}
    return out


def load_exp3():
    rows = []
    with open(PASQAL_CSV, newline="") as f:
        for r in csv.DictReader(f):
            form = normalize_formulation(r["formulation"])
            if not keep_formulation(form) or form not in FORMS:
                continue
            e = safe(parse_err(r["err_rydberg"]))
            if e is None:
                continue
            rows.append((r["L_type"], form, e))
    buckets = defaultdict(lambda: defaultdict(list))
    for lt, form, e in rows:
        buckets[lt][form].append(e)
    out = {}
    for lt in L_ORDER:
        out[lt] = {f: median(buckets[lt][f]) for f in FORMS if buckets[lt][f]}
    return out


def winner(scores: dict, forms=None):
    forms = forms or list(scores.keys())
    cands = [(scores[f], f) for f in forms if f in scores]
    if not cands:
        return None, None
    e, f = min(cands, key=lambda t: t[0])
    return f, e


# ── Figures ───────────────────────────────────────────────────

def fig_bars_by_regime(datasets):
    """3×3: rows=experiments, cols=L regimes ; bars=formulations."""
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.2), sharey=False)
    x = np.arange(len(FORMS))
    titles_row = [
        "Exp. 1 — $Q$ aléatoire",
        "Exp. 2 — $Q$ synthétique exacte",
        "Exp. 3 — Pasqal",
    ]

    for i, (name, data) in enumerate(datasets):
        for j, lt in enumerate(L_ORDER):
            ax = axes[i, j]
            scores = data.get(lt, {})
            vals = [scores.get(f, np.nan) for f in FORMS]
            cols = [COLORS[f] for f in FORMS]
            bars = ax.bar(x, vals, color=cols, edgecolor="white", lw=0.4)
            ax.set_yscale("log")
            ax.set_xticks(x)
            ax.set_xticklabels([FORM_SHORT[f] for f in FORMS], rotation=40, ha="right")
            if i == 0:
                ax.set_title(L_SHORT[lt], fontsize=12)
            if j == 0:
                ax.set_ylabel(r"médiane $\mathcal{E}_{\mathrm{Ryd}}$")
            # highlight best overall and best NLP
            if any(math.isfinite(v) for v in vals):
                ib = int(np.nanargmin(vals))
                bars[ib].set_edgecolor("black")
                bars[ib].set_linewidth(1.5)
            nlp_vals = [scores.get(f, np.nan) for f in NLP_FORMS]
            if any(math.isfinite(v) for v in nlp_vals):
                fbest, _ = winner(scores, NLP_FORMS)
                if fbest and fbest != FORMS[ib] if any(math.isfinite(v) for v in vals) else True:
                    idx = FORMS.index(fbest)
                    bars[idx].set_hatch("///")
                    bars[idx].set_edgecolor("#1c2833")
                    bars[idx].set_linewidth(1.0)
        axes[i, 0].annotate(
            titles_row[i],
            xy=(0, 0.5),
            xycoords=("axes fraction", "axes fraction"),
            xytext=(-0.28, 0.5),
            textcoords=("axes fraction", "axes fraction"),
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    fig.legend(
        handles=[Patch(facecolor=COLORS[f], label=FORM_SHORT[f]) for f in FORMS],
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        r"Meilleure formulation par régime $L$ (barre encadrée = meilleur global ; "
        r"hachures = meilleur NLP hors MDS)",
        y=1.05,
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGDIR / "lregime_bars_3exp.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lregime_bars_3exp.png")
    plt.close(fig)
    print("wrote", out)


def fig_winners(datasets):
    """Horizontal summary: winner + Eryd for each exp × L."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: best overall (incl. MDS)
    # Right: best NLP only
    scenarios = []
    for exp_name, data in datasets:
        for lt in L_ORDER:
            scenarios.append((exp_name, lt, data.get(lt, {})))

    def draw(ax, forms_pool, title):
        y = np.arange(len(scenarios))
        errs, names, cols, labels = [], [], [], []
        for exp_name, lt, scores in scenarios:
            f, e = winner(scores, forms_pool)
            errs.append(e if e is not None else np.nan)
            names.append(FORM_SHORT.get(f, "—") if f else "—")
            cols.append(COLORS.get(f, "#bdc3c7") if f else "#bdc3c7")
            labels.append(f"{exp_name}\n{L_SHORT[lt]}")
        ax.barh(y, errs, color=cols, height=0.7, edgecolor="white")
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(r"médiane $\mathcal{E}_{\mathrm{Ryd}}$")
        ax.set_title(title)
        for yi, (e, name) in enumerate(zip(errs, names)):
            if math.isfinite(e):
                ax.text(e * 1.5, yi, name, va="center", fontsize=8, fontweight="bold")

    draw(axes[0], FORMS, "Gagnant global (MDS inclus)")
    draw(axes[1], NLP_FORMS, "Gagnant NLP (hors MDS)")
    axes[1].legend(
        handles=[Patch(facecolor=COLORS[f], label=FORM_SHORT[f]) for f in FORMS],
        loc="lower right",
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        r"Formulation optimale — régimes $L$ fixe / adaptatif / optimal",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGDIR / "lregime_winners.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lregime_winners.png")
    plt.close(fig)
    print("wrote", out)


def fig_nlp_heatmap(datasets):
    """Heatmap log10(Eryd) for NLP forms × L regimes, one panel per exp."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    forms = NLP_FORMS
    for ax, (exp_name, data) in zip(axes, datasets):
        M = np.full((len(forms), len(L_ORDER)), np.nan)
        for j, lt in enumerate(L_ORDER):
            for i, f in enumerate(forms):
                if f in data.get(lt, {}):
                    M[i, j] = math.log10(data[lt][f])
        im = ax.imshow(M, aspect="auto", cmap="YlOrBr_r")
        ax.set_xticks(range(len(L_ORDER)))
        ax.set_xticklabels([L_SHORT[lt] for lt in L_ORDER])
        ax.set_yticks(range(len(forms)))
        ax.set_yticklabels([FORM_SHORT[f] for f in forms])
        ax.set_title(exp_name)
        # mark best per column
        for j in range(len(L_ORDER)):
            col = M[:, j]
            if np.any(np.isfinite(col)):
                i = int(np.nanargmin(col))
                ax.scatter(j, i, marker="*", s=120, c="#1a5276", zorder=5)
                ax.text(
                    j,
                    i,
                    f"\n{M[i, j]:.1f}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="#1a5276",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}\mathcal{E}_{\mathrm{Ryd}}$")
    fig.suptitle(
        r"Carte NLP (hors MDS) — étoile = meilleure formulation du régime $L$",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGDIR / "lregime_nlp_heatmap.pdf"
    fig.savefig(out)
    fig.savefig(FIGDIR / "lregime_nlp_heatmap.png")
    plt.close(fig)
    print("wrote", out)


def print_winners(datasets):
    print("\n=== Gagnants ===")
    for exp_name, data in datasets:
        print(f"\n{exp_name}")
        for lt in L_ORDER:
            fg, eg = winner(data.get(lt, {}), FORMS)
            fn, en = winner(data.get(lt, {}), NLP_FORMS)
            print(
                f"  {L_SHORT[lt]:10s}  global={FORM_SHORT.get(fg,'—'):10s} ({eg:.2e})  "
                f"NLP={FORM_SHORT.get(fn,'—'):10s} ({en:.2e})"
            )


def main():
    style()
    exp1 = load_exp1()
    exp2 = load_exp2()
    exp3 = load_exp3()
    datasets = [
        ("Exp. 1 — Q aléatoire", exp1),
        ("Exp. 2 — Q synthétique", exp2),
        ("Exp. 3 — Pasqal", exp3),
    ]
    print_winners(datasets)
    fig_bars_by_regime(datasets)
    fig_winners(datasets)
    fig_nlp_heatmap(datasets)
    print("done →", FIGDIR)


if __name__ == "__main__":
    main()
