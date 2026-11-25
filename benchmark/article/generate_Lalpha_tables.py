#!/usr/bin/env python3
"""Generate LaTeX tables for L-variable + α·L experiments from CSV."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = Path(__file__).resolve().parent
CSV_PASQAL = ROOT / "test_real" / "results" / "exp_L_alpha_full_results.csv"
CSV_EXTENDED = ROOT / "test_real" / "results" / "exp_L_alpha_full_results_n4_50.csv"
OUT = ARTICLE / "tables"

FORM_ORDER = ["Standard", "Simple", "Simple_square", "Taylor", "Taylor_square"]
FORM_LABEL = {
    "Standard": "Standard",
    "Simple": "Simple",
    "Simple_square": "Simple$^2$",
    "Taylor": "Taylor",
    "Taylor_square": "Taylor$^2$",
}
SOURCE_LABEL = {
    "exp3_pasqal": "Pasqal (GML)",
    "exp2_Q_exacte_synthetique": "$Q$ exacte synthétique",
    "exp1_Q_aleatoire": "$Q$ aléatoire",
}
# Alphas shown in main tables (keep compact)
ALPHAS_MAIN = [1e-6, 1e-4, 1e-2, 1.0]
ALPHAS_ALL = [1e-6, 1e-4, 1e-2, 1.0, 5.0, 10.0]


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def fmt_sci(x, bold=False):
    if x is None or not math.isfinite(x):
        s = "---"
    elif x == 0:
        s = "0"
    else:
        exp = int(math.floor(math.log10(abs(x))))
        mant = x / (10**exp)
        if abs(mant - 1.0) < 1e-3:
            s = f"10^{{{exp}}}"
        else:
            s = f"{mant:.2f}\\times10^{{{exp}}}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def fmt_alpha(a):
    if a >= 1:
        return f"${a:g}$"
    exp = int(round(math.log10(a)))
    return f"$10^{{{exp}}}$"


def fmt_pct(m, n):
    if n == 0:
        return "---"
    return f"{100.0 * m / n:.0f}\\%"


def fmt_L(x):
    if x is None or not math.isfinite(x):
        return "---"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def load_rows(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_all_rows():
    rows = []
    with CSV_PASQAL.open(newline="") as f:
        rows.extend(
            r for r in csv.DictReader(f) if r.get("source") == "exp3_pasqal"
        )
    with CSV_EXTENDED.open(newline="") as f:
        rows.extend(
            r
            for r in csv.DictReader(f)
            if r.get("source") in {"exp2_Q_exacte_synthetique", "exp1_Q_aleatoire"}
        )
    if not rows:
        raise RuntimeError("Aucune ligne L-α chargée depuis les CSV de résultats")
    return rows


def group_stats(rows):
    """Key: (source, use_zmin bool, formulation, alpha) -> stats dict."""
    buckets = defaultdict(list)
    for r in rows:
        key = (
            r["source"],
            r["use_zmin"] == "true",
            r["formulation"],
            float(r["alpha"]),
        )
        buckets[key].append(r)

    out = {}
    for key, g in buckets.items():
        matches = sum(1 for r in g if r["dmin_matches_zmin"] == "true")
        rys = [fnum(r["err_rydberg"]) for r in g]
        Ls = [fnum(r["L_found"]) for r in g]
        ratios = [fnum(r["ratio_dmin_zmin"]) for r in g]
        rys = [x for x in rys if x is not None]
        Ls = [x for x in Ls if x is not None]
        ratios = [x for x in ratios if x is not None]
        out[key] = {
            "n": len(g),
            "match": matches,
            "ry_med": median(rys) if rys else None,
            "ry_mean": mean(rys) if rys else None,
            "L_mean": mean(Ls) if Ls else None,
            "ratio_mean": mean(ratios) if ratios else None,
        }
    return out


def source_note(rows, source):
    subset = [r for r in rows if r["source"] == source]
    if source == "exp3_pasqal":
        return "48 fichiers GML, $n=25$"

    instances = sorted({str(r["instance"]) for r in subset})
    ns = sorted({int(float(r["n"])) for r in subset})
    inst_label = "instance" if len(instances) == 1 else "instances"
    return f"{len(instances)} {inst_label}, $n\\in\\{{{','.join(str(n) for n in ns)}\\}}$"


def closest_alpha(a, alphas=ALPHAS_ALL):
    return min(alphas, key=lambda t: abs(t - a))


def generate_overview(stats, rows):
    """One compact overview: match rate + median Eryd for α=1e-4, avec/sans."""
    pasqal_note = source_note(rows, "exp3_pasqal")
    synth_note = source_note(rows, "exp2_Q_exacte_synthetique")
    rand_note = source_note(rows, "exp1_Q_aleatoire")
    lines = [
        "% Auto-generated — L-variable + α·L overview",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Expérience~$L$ variable : pénalité $\\alpha L$ et vérification "
        "$d_{\\min}(\\hat{\\mathbf{x}}) \\approx z_{\\min}^{\\mathrm{opt}} = "
        "(\\cSix/Q_{\\max})^{1/6}$. "
        "Taux de concordance ($|d_{\\min}-z_{\\min}|/z_{\\min}<0{,}2$) et "
        "$\\mathcal{E}_{\\mathrm{Ryd}}$ médiane pour $\\alpha=10^{-4}$. "
        f"Pasqal : {pasqal_note} ; synthétique : {synth_note} ; aléatoire : {rand_note}.}}",
        "\\label{tab:lalpha-overview}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{@{}llcccc@{}}",
        "  \\toprule",
        "  Source & Formulation & "
        "\\multicolumn{2}{c}{avec $z_{\\min}$} & "
        "\\multicolumn{2}{c}{sans $z_{\\min}$} \\\\",
        "  \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "   &  & Concord. & $\\mathcal{E}_{\\mathrm{Ryd}}$ & "
        "Concord. & $\\mathcal{E}_{\\mathrm{Ryd}}$ \\\\",
        "  \\midrule",
    ]
    alpha = 1e-4
    for source in ["exp3_pasqal", "exp2_Q_exacte_synthetique", "exp1_Q_aleatoire"]:
        first = True
        for form in FORM_ORDER:
            sa = stats.get((source, True, form, alpha))
            sn = stats.get((source, False, form, alpha))
            if sa is None and sn is None:
                continue
            src = SOURCE_LABEL[source] if first else ""
            first = False
            # bold best median ry among row (avec and sans)
            cands = []
            if sa and sa["ry_med"] is not None:
                cands.append(sa["ry_med"])
            if sn and sn["ry_med"] is not None:
                cands.append(sn["ry_med"])
            best = min(cands) if cands else None

            def cell(st):
                if st is None:
                    return "---", "---"
                bold = best is not None and st["ry_med"] == best
                return fmt_pct(st["match"], st["n"]), fmt_sci(st["ry_med"], bold=bold)

            pa, ea = cell(sa)
            pn, en = cell(sn)
            lines.append(
                f"  {src} & {FORM_LABEL[form]} & {pa} & {ea} & {pn} & {en} \\\\"
            )
        lines.append("  \\midrule")
    if lines[-1] == "  \\midrule":
        lines[-1] = "  \\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def generate_detail(stats, rows, source: str, alphas=ALPHAS_MAIN, label_suffix=""):
    """Detail table for one source: formulation × α × avec/sans."""
    lab = {
        "exp3_pasqal": "pasqal",
        "exp2_Q_exacte_synthetique": "synth",
        "exp1_Q_aleatoire": "rand",
    }[source]
    n_note = source_note(rows, source)
    lines = [
        f"% Auto-generated — L-α detail ({source})",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{"
        f"{SOURCE_LABEL[source]} ({n_note}) : $L$ variable, objectif $f(\\mathbf{{x}})+\\alpha L$. "
        "Concordance $d_{\\min}\\approx z_{\\min}^{\\mathrm{opt}}$, "
        "$\\bar L$ et $\\mathcal{E}_{\\mathrm{Ryd}}$ médiane.}",
        f"\\label{{tab:lalpha-{lab}{label_suffix}}}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{@{}llrrcrrc@{}}",
        "  \\toprule",
        "  Form. & $\\alpha$ & "
        "\\multicolumn{3}{c}{avec $z_{\\min}$} & "
        "\\multicolumn{3}{c}{sans $z_{\\min}$} \\\\",
        "  \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "   &  & Conc. & $\\bar L$ & $\\mathcal{E}_{\\mathrm{Ryd}}$ & "
        "Conc. & $\\bar L$ & $\\mathcal{E}_{\\mathrm{Ryd}}$ \\\\",
        "  \\midrule",
    ]
    for form in FORM_ORDER:
        for i, a in enumerate(alphas):
            sa = stats.get((source, True, form, a))
            sn = stats.get((source, False, form, a))
            form_lab = FORM_LABEL[form] if i == 0 else ""
            if sa is None and sn is None:
                continue

            def triple(st):
                if st is None:
                    return "---", "---", "---"
                return (
                    fmt_pct(st["match"], st["n"]),
                    fmt_L(st["L_mean"]),
                    fmt_sci(st["ry_med"]),
                )

            ca, La, ea = triple(sa)
            cn, Ln, en = triple(sn)
            lines.append(
                f"  {form_lab} & {fmt_alpha(a)} & {ca} & {La} & {ea} & {cn} & {Ln} & {en} \\\\"
            )
        lines.append("  \\midrule")
    if lines[-1] == "  \\midrule":
        lines[-1] = "  \\bottomrule"
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def generate_scaling_note(rows):
    """Short table: how often L_chosen < L* triggered scaling."""
    n = len(rows)
    scaled = [r for r in rows if r["scale_applied"] == "true"]
    ns = len(scaled)
    if scaled:
        lams = [fnum(r["lambda"]) for r in scaled]
        lams = [x for x in lams if x is not None]
        lam_med = median(lams) if lams else None
        match_before = sum(1 for r in scaled if r["dmin_matches_zmin"] == "true")
        match_after = sum(1 for r in scaled if r["dmin_matches_zmin_scaled"] == "true")
    else:
        lam_med = None
        match_before = 0
        match_after = 0
    lines = [
        "% Auto-generated — scaling summary",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Scaling cohérent vers $L_{\\mathrm{choisi}}=10$ : "
        "si $L^{\\star}>L_{\\mathrm{choisi}}$ on applique "
        "$\\lambda=L_{\\mathrm{choisi}}/L^{\\star}$, "
        "$\\mathbf{x}'=\\lambda\\mathbf{x}^{\\star}$, "
        "$z_{\\min}'=\\lambda z_{\\min}$, $Q'=Q/\\lambda^{6}$ ; "
        "sinon aucune transformation.}",
        "\\label{tab:lalpha-scaling}",
        "\\footnotesize",
        "\\begin{tabular}{@{}lrl@{}}",
        "  \\toprule",
        "  Quantité & Valeur & Commentaire \\\\",
        "  \\midrule",
        f"  Runs totaux & {n} & 3 sources $\\times$ formulations $\\times$ $\\alpha$ \\\\",
        f"  Scaling appliqué & {ns} ({100*ns/n:.1f}\\%) & $L^{{\\star}}>L_{{\\mathrm{{choisi}}}}$ \\\\",
        f"  $\\lambda$ médian (si scaling) & "
        f"{fmt_sci(lam_med) if lam_med is not None else '---'} & "
        f"$L_{{\\mathrm{{choisi}}}}/L^{{\\star}}$ \\\\",
        f"  Concord. $d_{{\\min}}\\approx z_{{\\min}}$ (avant) & "
        f"{fmt_pct(match_before, ns) if ns else '---'} & avant scaling \\\\",
        f"  Concord. $d_{{\\min}}'\\approx z_{{\\min}}'$ (après) & "
        f"{fmt_pct(match_after, ns) if ns else '---'} & après scaling \\\\",
        "  \\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    for path in (CSV_PASQAL, CSV_EXTENDED):
        if not path.exists():
            raise SystemExit(f"CSV introuvable : {path}")

    rows = load_all_rows()
    stats = group_stats(rows)
    # Remap keys to exact ALPHAS_ALL floats
    stats2 = {}
    for (src, uz, form, a), st in stats.items():
        stats2[(src, uz, form, closest_alpha(a))] = st

    OUT.mkdir(exist_ok=True)
    (OUT / "exp_lalpha_overview.tex").write_text(
        generate_overview(stats2, rows), encoding="utf-8"
    )
    (OUT / "exp_lalpha_pasqal.tex").write_text(
        generate_detail(stats2, rows, "exp3_pasqal"), encoding="utf-8"
    )
    (OUT / "exp_lalpha_synth.tex").write_text(
        generate_detail(stats2, rows, "exp2_Q_exacte_synthetique"), encoding="utf-8"
    )
    (OUT / "exp_lalpha_random.tex").write_text(
        generate_detail(stats2, rows, "exp1_Q_aleatoire"), encoding="utf-8"
    )
    (OUT / "exp_lalpha_scaling.tex").write_text(
        generate_scaling_note(rows), encoding="utf-8"
    )
    print(
        "Generated L-α tables from "
        f"{CSV_PASQAL.name} + {CSV_EXTENDED.name} ({len(rows)} rows) → {OUT}/"
    )


if __name__ == "__main__":
    main()
