#!/usr/bin/env python3
"""Generate LaTeX result tables for article.tex from benchmark data."""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
ARTICLE = Path(__file__).resolve().parent
PASQAL_CSV = ROOT / "test_real" / "pasqal_gml_benchmark.csv"
EXP2_LOG = ARTICLE / "exp2_benchmark.log"

FORM_ORDER = [
    "MDS_Exact", "Standard", "Taylor", "Taylor_square", "Simple_square", "Simple",
]
FORM_LABEL = {
    "MDS_Exact": "MDS exact",
    "Standard": "Standard",
    "Taylor": "Taylor",
    "Taylor_square": "Taylor$^2$",
    "Simple_square": "Simple$^2$",
    "Simple": "Simple",
}
L_LABEL = {
    "L_fixe": "Fixe ($L{=}10$)",
    "L_adaptatif": "Adaptatif ($L{=}100\\sqrt{n}$)",
    "L_optimal": "Optimal ($L{\\approx}r_{\\min}$)",
}
L_LABEL_SHORT = {
    "L_fixe": "Fixe",
    "L_adaptatif": "Adapt.",
    "L_optimal": "Optim.",
}
STATUS_SHORT = {
    "LOCALLY_SOLVED": "loc.",
    "ALMOST_LOCALLY_SOLVED": "almost",
    "ALMOST_LOCALLY_SOLVE": "almost",
    "ITERATION_LIMIT": "iter.\\,lim.",
    "SLOW_PROGRESS": "slow",
    "ECHEC": "échec",
    "ÉCHEC": "échec",
    "ÉCHE": "échec",
    "MDS_EXACT": "MDS",
    "EXACT": "MDS",
}
STRATEGY_SHORT = {
    "exact": "MDS",
    "single": "sng.",
    "multistart": "ms.",
}
GEOM_LABEL = {
    "freespace": "Espace libre",
    "sqauare": "Grille carrée",
    "triangle": "Grille triangulaire",
}

# Exp. 1 — moyennes sur 10 instances (Q_bruit_dist, err_bruit)
EXP1_SUMMARY = {
    (4, "L_fixe"): {
        "MDS_Exact": (6.7853e10, 0.2400),
        "Standard": (7.2295e10, 1.0375),
        "Taylor": (6.5066e10, 0.2780),
        "Taylor_square": (6.5066e10, 0.4187),
        "Simple_square": (6.7853e10, 0.1329),
        "Simple": (6.7852e10, 0.1272),
    },
    (4, "L_adaptatif"): {
        "MDS_Exact": (1.3001e-11, 0.0001),
        "Standard": (1.3013e-11, 0.0106),
        "Taylor": (2.5619e-14, 0.0124),
        "Taylor_square": (1.6747e-15, 0.6196),
        "Simple_square": (1.6868e-11, 0.0867),
        "Simple": (1.2996e-11, 0.0422),
    },
    (4, "L_optimal"): {
        "MDS_Exact": (4.4182e8, 0.0001),
        "Standard": (2.0622e8, 0.1881),
        "Taylor": (2.0622e8, 1.4044),
        "Taylor_square": (2.0622e8, 3.2804),
        "Simple_square": (4.4182e8, 0.4960),
        "Simple": (4.4179e8, 0.0506),
    },
    (10, "L_fixe"): {
        "MDS_Exact": (7.5127e29, 0.0002),
        "Standard": (7.5127e29, 4.3028),
        "Taylor": (7.5127e29, 9.3116),
        "Taylor_square": (7.5127e29, 7.6777),
        "Simple_square": (7.5127e29, 0.2393),
        "Simple": (7.5127e29, 0.2158),
    },
    (10, "L_adaptatif"): {
        "MDS_Exact": (9.8626e6, 0.0003),
        "Standard": (9.8625e6, 0.1480),
        "Taylor": (8.1488e-10, 2.0453),
        "Taylor_square": (1.0736e-9, 3.4030),
        "Simple_square": (9.8626e6, 0.6497),
        "Simple": (9.8626e6, 0.3416),
    },
    (10, "L_optimal"): {
        "MDS_Exact": (3.4851e14, 0.0002),
        "Standard": (3.8702e14, 3.8775),
        "Taylor": (3.4832e14, 4.3872),
        "Taylor_square": (3.4832e14, 5.6665),
        "Simple_square": (3.4851e14, 0.2192),
        "Simple": (3.4851e14, 0.1966),
    },
    (20, "L_fixe"): {
        "MDS_Exact": (2.6607e26, 0.0005),
        "Standard": (2.6607e26, 15.8143),
        "Taylor": (2.6607e26, 39.3839),
        "Taylor_square": (2.6607e26, 26.1260),
        "Simple_square": (2.6607e26, 0.7962),
        "Simple": (2.6607e26, 0.7163),
    },
    (20, "L_adaptatif"): {
        "MDS_Exact": (3.2639e6, 0.0007),
        "Standard": (3.2639e6, 0.1489),
        "Taylor": (3.2321e-6, 13.0560),
        "Taylor_square": (4.7857e-5, 43.4510),
        "Simple_square": (3.2639e6, 2.8041),
        "Simple": (3.2639e6, 1.5204),
    },
    (20, "L_optimal"): {
        "MDS_Exact": (5.3168e27, 0.0007),
        "Standard": (6.6460e27, 15.7779),
        "Taylor": (5.3168e27, 47.9854),
        "Taylor_square": (5.3168e27, 30.8307),
        "Simple_square": (5.3168e27, 0.8445),
        "Simple": (5.3168e27, 0.7134),
    },
    (30, "L_fixe"): {
        "MDS_Exact": (9.0921e28, 0.0006),
        "Standard": (9.0921e28, 74.6353),
        "Taylor": (1.0102e29, 308.8839),
        "Taylor_square": (9.0921e28, 72.7448),
        "Simple_square": (9.0921e28, 2.2318),
        "Simple": (9.0921e28, 1.4558),
    },
    (30, "L_adaptatif"): {
        "MDS_Exact": (7.2511e15, 0.0007),
        "Standard": (7.2511e15, 0.4217),
        "Taylor": (7.2503e15, 21.7532),
        "Taylor_square": (7.2503e15, 42.7168),
        "Simple_square": (7.2511e15, 15.6652),
        "Simple": (7.2511e15, 3.7851),
    },
    (30, "L_optimal"): {
        "MDS_Exact": (2.4388e37, 0.0008),
        "Standard": (3.4839e37, 91.1997),
        "Taylor": (2.4388e37, 788.5601),
        "Taylor_square": (2.4388e37, 72.2518),
        "Simple_square": (2.4388e37, 2.1722),
        "Simple": (2.4388e37, 1.6584),
    },
    (50, "L_fixe"): {
        "MDS_Exact": (5.9710e41, 0.0057),
        "Standard": (4.0612e31, 460.5037),
        "Taylor": (5.9710e41, 16991.0516),
        "Taylor_square": (5.9710e41, 2083.4560),
        "Simple_square": (5.9710e41, 11.1830),
        "Simple": (5.9710e41, 7.8735),
    },
    (50, "L_adaptatif"): {
        "MDS_Exact": (7.8221e27, 0.0025),
        "Standard": (7.8221e27, 670.6686),
        "Taylor": (7.8221e27, 2233.0776),
        "Taylor_square": (8.6912e27, 259.8940),
        "Simple_square": (7.8221e27, 93.2997),
        "Simple": (7.8221e27, 24.0149),
    },
    (50, "L_optimal"): {
        "MDS_Exact": (6.7010e33, 0.0012),
        "Standard": (8.3762e33, 405.5371),
        "Taylor": (6.7010e33, 16513.2991),
        "Taylor_square": (6.7010e33, 293.0732),
        "Simple_square": (6.7010e33, 11.6982),
        "Simple": (6.7010e33, 8.6196),
    },
}


def fmt_sci(x, bold=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "$\\mathbf{\\infty}$" if bold else "$\\infty$"
    if x == 0:
        return "$\\mathbf{0}$" if bold else "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    if abs(mant - round(mant)) < 0.06:
        inner = f"{int(round(mant))}\\times10^{{{exp}}}"
    else:
        inner = f"{mant:.2f}\\times10^{{{exp}}}"
    if bold:
        return f"$\\mathbf{{{inner}}}$"
    return f"${inner}$"


def normalize_formulation(name):
    aliases = {
        "MDS_Exact": "MDS_Exact",
        "Taylor_square": "Taylor_square",
        "Simple_square": "Simple_square",
    }
    return aliases.get(name, name)


KEEP_FORMULATIONS = set(FORM_ORDER)


def keep_formulation(name):
    return normalize_formulation(name) in KEEP_FORMULATIONS



def short_fname(file):
    m = re.search(r"([\d.]+)", file)
    if m:
        num = m.group(1).rstrip(".")
        try:
            return f"w={float(num):.3g}"
        except ValueError:
            pass
    base = file.replace(".gml", "").replace("_", "\\_")
    return base[:14] + ("\\ldots" if len(base) > 14 else "")


def fmt_status(s, short=False):
    if not s:
        return "---"
    if short and s in STATUS_SHORT:
        return STATUS_SHORT[s]
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def form_label(name):
    return FORM_LABEL.get(normalize_formulation(name), name)


def fmt_rmse(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    if isinstance(x, float) and math.isinf(x):
        return "$\\infty$"
    ax = abs(x)
    if ax == 0:
        return "$0$"
    if ax < 1e-2 or ax >= 100:
        return fmt_sci(x)
    return f"${x:.3g}$"


def fmt_time(t):
    if t < 0.01:
        return f"{t*1000:.0f}\\,ms"
    if t < 1:
        return f"{t:.2f}\\,s"
    if t < 60:
        return f"{t:.1f}\\,s"
    if t < 3600:
        return f"{t/60:.1f}\\,min"
    return f"{t/3600:.1f}\\,h"


def best_form(data):
    finite = [(f, data[f][0]) for f in FORM_ORDER if f in data and math.isfinite(data[f][0])]
    if not finite:
        return None
    return min(finite, key=lambda x: x[1])[0]


def best_exp2_row(rows):
    """Best (formulation, strategy) by obj_rydberg for one (n, L_type) block."""
    finite = [
        r for r in rows
        if math.isfinite(r.get("obj_rydberg", float("inf")))
    ]
    if not finite:
        return None
    return min(finite, key=lambda r: r["obj_rydberg"])


def best_exp2_nlp(rows):
    finite = [
        r for r in rows
        if r.get("strategy") != "exact"
        and math.isfinite(r.get("obj_rydberg", float("inf")))
    ]
    if not finite:
        return None
    return min(finite, key=lambda r: r["obj_rydberg"])


def generate_exp1_main():
    lines = []
    lines.append("% Auto-generated — Exp. 1 main summary (best method per config)")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Expérience~1 ($Q_{\\mathrm{bruit\\,dist}}$) : meilleure formulation par $(n,L)$ "
                 "(moyenne sur 10 instances). Métrique : $\\mathcal{E}_{\\mathrm{Ryd}}$ vs.\\ $Q$ aléatoire.}")
    lines.append("\\label{tab:exp1-best}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}ccclrr@{}}")
    lines.append("  \\toprule")
    lines.append("  $n$ & Régime $L$ & Meilleure & $\\mathcal{E}_{\\mathrm{Ryd}}$ & Temps & $\\mathcal{E}_{\\mathrm{Ryd}}$ MDS \\\\")
    lines.append("  \\midrule")
    for n in [4, 10, 20, 30, 50]:
        for lt in ["L_fixe", "L_adaptatif", "L_optimal"]:
            data = EXP1_SUMMARY[(n, lt)]
            bf = best_form(data)
            be, bt = data[bf]
            me, mt = data["MDS_Exact"]
            lines.append(
                f"  {n} & {L_LABEL[lt]} & {FORM_LABEL[bf]} & {fmt_sci(be)} & {fmt_time(bt)} & {fmt_sci(me)} \\\\"
            )
    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def generate_exp1_appendix():
    lines = []
    lines.append("% Auto-generated — Exp. 1 full summary tables")
    lines.append("\\section{Résultats détaillés --- Expérience~1 ($Q_{\\mathrm{bruit\\,dist}}$)}")
    lines.append("\\label{app:exp1}")
    lines.append("")
    lines.append("Matrice $Q$ générée aléatoirement via \\texttt{Q\\_bruit\\_dist} ; "
                 "objectif : retrouver $\\mathbf{x}$ minimisant $\\mathcal{E}_{\\mathrm{Ryd}}$. "
                 "Moyennes sur 10 instances indépendantes.")
    lines.append("")
    for n in [4, 10, 20, 30, 50]:
        for lt in ["L_fixe", "L_adaptatif", "L_optimal"]:
            data = EXP1_SUMMARY[(n, lt)]
            bf = best_form(data)
            lines.append(f"\\subsection{{$n={n}$, {L_LABEL[lt]}}}")
            lines.append("\\begin{table}[H]")
            lines.append("\\centering")
            lines.append("\\small")
            lines.append("\\begin{tabular}{@{}lrr@{}}")
            lines.append("  \\toprule")
            lines.append("  Formulation & $\\mathcal{E}_{\\mathrm{Ryd}}$ & Temps moy. \\\\")
            lines.append("  \\midrule")
            for f in FORM_ORDER:
                e, t = data[f]
                label = FORM_LABEL[f]
                e_fmt = fmt_sci(e, bold=(f == bf))
                lines.append(f"  {label} & {e_fmt} & {fmt_time(t)} \\\\")
            lines.append("  \\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("\\end{table}")
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_err(s):
    try:
        v = float(s)
        return v
    except ValueError:
        return float("inf")


def load_pasqal():
    rows = []
    with open(PASQAL_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if keep_formulation(r["formulation"]):
                rows.append(r)
    return rows


def generate_exp3_main(rows):
    lines = []
    lines.append("% Auto-generated — Exp. 3 Pasqal summary")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Expérience~3 (instances Pasqal, $n=25$, $Q$ exact) : "
                 "$\\mathcal{E}_{\\mathrm{Ryd}}$ médiane et temps moyen par géométrie et régime $L$. "
                 "16 fichiers GML par géométrie. "
                 "Légende $L$ : Fixe ($L{=}10$), Adapt.\\ ($L{=}100\\sqrt{n}$), "
                 "Optim.\\ ($L{\\approx}r_{\\min}$).}")
    lines.append("\\label{tab:exp3-pasqal}")
    lines.append("\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\begin{tabular}{@{}lllrr@{}}")
    lines.append("  \\toprule")
    lines.append("  Géométrie & $L$ & Formulation & $\\mathcal{E}_{\\mathrm{Ryd}}$ (méd.) & Temps \\\\")
    lines.append("  \\midrule")

    for folder in ["freespace", "sqauare", "triangle"]:
        for lt in ["L_fixe", "L_adaptatif", "L_optimal"]:
            agg = defaultdict(list)
            times = defaultdict(list)
            for r in rows:
                if r["folder"] != folder or r["L_type"] != lt:
                    continue
                f = normalize_formulation(r["formulation"])
                e = parse_err(r["err_rydberg"])
                if math.isfinite(e):
                    agg[f].append(e)
                times[f].append(float(r["time_s"]))
            ranked = []
            for f in FORM_ORDER:
                if f not in agg and f not in times:
                    continue
                errs = sorted(agg[f]) if agg[f] else [float("inf")]
                med = errs[len(errs) // 2]
                ranked.append((med, f, mean(times[f])))
            ranked.sort()
            best_med = ranked[0][0]
            show = ranked[: min(4, len(ranked))]
            nrows = len(show)
            for i, (med, f, t) in enumerate(show):
                e_fmt = fmt_sci(med, bold=(med == best_med))
                if i == 0:
                    lines.append(
                        f"  \\multirow{{{nrows}}}{{*}}{{{GEOM_LABEL[folder]}}} & "
                        f"\\multirow{{{nrows}}}{{*}}{{{L_LABEL_SHORT[lt]}}} & "
                        f"{FORM_LABEL[f]} & {e_fmt} & {fmt_time(t)} \\\\"
                    )
                else:
                    lines.append(f"  & & {FORM_LABEL[f]} & {e_fmt} & {fmt_time(t)} \\\\")
            lines.append("  \\midrule")
    lines[-1] = "  \\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def generate_exp3_appendix(rows):
    lines = []
    lines.append("% Auto-generated — Exp. 3 Pasqal full results")
    lines.append("\\section{Résultats détaillés --- Expérience~3 (Pasqal GML, $Q$ exact)}")
    lines.append("\\label{app:exp3}")
    lines.append("")
    lines.append("Instances fournies par Pasqal : matrices $Q$ exactes sur graphes "
                 "d'interaction ($n=25$ atomes, 300 arêtes), trois géométries de réseau. "
                 "Fichier source : \\texttt{pasqal\\_gml\\_benchmark.csv}. "
                 "Les noms de fichiers sont abrégés (\\texttt{w=\\dots} = poids du graphe). "
                 "Statuts : loc.\\ = \\texttt{LOCALLY\\_SOLVED}, almost = "
                 "\\texttt{ALMOST\\_LOCALLY\\_SOLVED}, iter.\\,lim.\\ = "
                 "\\texttt{ITERATION\\_LIMIT}.")
    lines.append("")

    colspec = (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{2.0cm}"
        "c"
        ">{\\raggedright\\arraybackslash}p{1.55cm}"
        "r"
        "c"
        ">{\\raggedright\\arraybackslash}p{1.5cm}"
        "@{}"
    )

    for folder in ["freespace", "sqauare", "triangle"]:
        sub_rows = [r for r in rows if r["folder"] == folder]
        if not sub_rows:
            continue
        lines.append(f"\\subsection{{{GEOM_LABEL[folder]}}}")
        lines.append("{\\footnotesize")
        lines.append("\\setlength{\\tabcolsep}{3pt}")
        lines.append("\\renewcommand{\\arraystretch}{1.05}")
        lines.append(f"\\begin{{longtable}}{{{colspec}}}")
        lines.append("  \\toprule")
        lines.append(
            "  Fichier & $L$ & Formulation & $\\mathcal{E}_{\\mathrm{Ryd}}$ & "
            "Temps & Statut \\\\"
        )
        lines.append("  \\midrule")
        lines.append("  \\endhead")
        lines.append("  \\midrule")
        lines.append(
            f"  \\multicolumn{{6}}{{r}}{{\\footnotesize\\textit{{suite --- {GEOM_LABEL[folder]}}}}} \\\\"
        )
        lines.append("  \\endfoot")
        for r in sub_rows:
            e = parse_err(r["err_rydberg"])
            lines.append(
                f"  {short_fname(r['file'])} & "
                f"{L_LABEL_SHORT.get(r['L_type'], r['L_type'])} & "
                f"{form_label(r['formulation'])} & "
                f"{fmt_sci(e)} & {fmt_time(float(r['time_s']))} & "
                f"{fmt_status(r['status'], short=True)} \\\\"
            )
        lines.append("  \\bottomrule")
        lines.append("\\end{longtable}")
        lines.append("}")
        lines.append("")

    return "\n".join(lines) + "\n"


def parse_exp2_summary(log_path):
    """Parse averaged results from Exp.~2 (Q exact) benchmark log."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    marker = "MOYENNES SUR"
    if marker not in text:
        return []
    text = text[text.index(marker):]
    rows = []
    n = None
    lt = None
    pat_header = re.compile(r"^\s*n=(\d+)\s+(L_\w+)\s*$")
    pat_row = re.compile(
        r"^\s+(\S+)\s+\[(\w+)\s*\]\s+obj_ry=([+\-]?[\d.eE+\-infINF]+|NaN)\s+"
        r"rmse=([+\-]?[\d.eE+\-infINF]+|NaN)\s+t_moy=\s*([\d.]+)s"
    )
    for line in text.splitlines():
        m = pat_header.match(line)
        if m:
            n, lt = int(m.group(1)), m.group(2)
            continue
        m = pat_row.match(line)
        if m and n is not None:
            form = normalize_formulation(m.group(1))
            if not keep_formulation(form):
                continue
            rows.append({
                "n": n,
                "L_type": lt,
                "formulation": form,
                "strategy": m.group(2),
                "obj_rydberg": parse_err(m.group(3)),
                "rmse_dist": parse_err(m.group(4)),
                "time_s": float(m.group(5)),
            })
    return rows


def parse_exp2_instances(log_path):
    """Parse per-instance rows from Exp.~2 (Q exact) benchmark log."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    if "MOYENNES SUR" in text:
        text = text[: text.index("MOYENNES SUR")]
    rows = []
    inst = None
    n = None
    lt = None
    lval = None
    strategy = None
    pat_block = re.compile(r"n=(\d+)\s+\|\s+(L_\w+)\s+\(L=([\d.]+)\)")
    pat_mds = re.compile(
        r"\[MDS_Exact\s*\] obj_rydberg=([+\-]?[\d.eE+\-infINF]+)\s+"
        r"rmse_dist=([+\-]?[\d.eE+\-infINF]+)\s+t=\s*([\d.]+)s"
    )
    pat_nlp = re.compile(
        r"\[(\w+)\s*\] obj_solver=([+\-]?[\d.eE+\-infINF]+)\s+"
        r"obj_rydberg=([+\-]?[\d.eE+\-infINF]+)\s+rmse_dist=([+\-]?[\d.eE+\-infINF]+)\s+"
        r"t=\s*([\d.]+)s\s+\[([^\]]+)\]"
    )
    for line in text.splitlines():
        m = re.search(r"Instance (\d+) / (\d+)", line)
        if m:
            inst = int(m.group(1))
            continue
        m = pat_block.search(line)
        if m:
            n, lt, lval = int(m.group(1)), m.group(2), float(m.group(3))
            strategy = None
            continue
        if "Stratégie : single" in line:
            strategy = "single"
            continue
        if "Stratégie : multistart" in line:
            strategy = "multistart"
            continue
        m = pat_mds.search(line)
        if m and inst is not None and n is not None:
            rows.append({
                "instance": inst, "n": n, "L_type": lt, "L_val": lval,
                "strategy": "exact", "formulation": "MDS_Exact",
                "obj_rydberg": parse_err(m.group(1)),
                "rmse_dist": parse_err(m.group(2)),
                "time_s": float(m.group(3)),
                "status": "MDS_EXACT",
            })
            continue
        m = pat_nlp.search(line)
        if m and inst is not None and n is not None and strategy:
            form = normalize_formulation(m.group(1))
            if not keep_formulation(form):
                continue
            rows.append({
                "instance": inst, "n": n, "L_type": lt, "L_val": lval,
                "strategy": strategy,
                "formulation": form,
                "obj_rydberg": parse_err(m.group(3)),
                "rmse_dist": parse_err(m.group(4)),
                "time_s": float(m.group(5)),
                "status": m.group(6).strip(),
            })
    return rows


def group_exp2_summary(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["n"], r["L_type"])].append(r)
    return grouped


def generate_exp2_main(summary_rows):
    grouped = group_exp2_summary(summary_rows)
    lines = []
    lines.append("% Auto-generated — Exp. 2 main summary (Q exacte synthétique)")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Expérience~2 ($Q$ exacte synthétique) : meilleure méthode par $(n,L)$ "
                 "(moyenne sur 10 instances). Légende $L$ : Fixe ($L{=}10$), Adapt.\\ "
                 "($L{=}100\\sqrt{n}$), Optim.\\ ($L{\\approx}r_{\\min}$).}")
    lines.append("\\label{tab:exp2-best}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.05}")
    lines.append("\\begin{tabular}{@{}ccclrrrrrl@{}}")
    lines.append("  \\toprule")
    lines.append("  $n$ & $L$ & Meilleure & St. & $\\mathcal{E}_{\\mathrm{Ryd}}$ & "
                 "RMSE & $t$ & $\\mathcal{E}_{\\mathrm{Ryd}}^{\\mathrm{MDS}}$ & "
                 "NLP & $\\mathcal{E}_{\\mathrm{Ryd}}^{\\mathrm{NLP}}$ \\\\")
    lines.append("  \\midrule")
    for n in sorted({k[0] for k in grouped}):
        for lt in ["L_fixe", "L_adaptatif", "L_optimal"]:
            block = grouped.get((n, lt))
            if not block:
                continue
            best = best_exp2_row(block)
            mds = next((r for r in block if r["strategy"] == "exact"), None)
            nlp = best_exp2_nlp(block)
            mds_e = mds["obj_rydberg"] if mds else float("inf")
            nlp_label = "---"
            nlp_e = float("inf")
            if nlp:
                nlp_label = form_label(nlp["formulation"])
                nlp_e = nlp["obj_rydberg"]
            lines.append(
                f"  {n} & {L_LABEL_SHORT[lt]} & {form_label(best['formulation'])} & "
                f"{STRATEGY_SHORT.get(best['strategy'], best['strategy'])} & "
                f"{fmt_sci(best['obj_rydberg'])} & {fmt_rmse(best['rmse_dist'])} & "
                f"{fmt_time(best['time_s'])} & {fmt_sci(mds_e)} & "
                f"{nlp_label} & {fmt_sci(nlp_e)} \\\\"
            )
    lines.append("  \\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def generate_exp2_appendix(summary_rows):
    grouped = group_exp2_summary(summary_rows)
    lines = []
    lines.append("% Auto-generated — Exp. 2 full summary tables (Q exacte)")
    lines.append("\\section{Résultats détaillés --- Expérience~2 ($Q$ exacte synthétique)}")
    lines.append("\\label{app:exp2}")
    lines.append("")
    lines.append("Positions $\\mathbf{x}^{\\mathrm{ref}}$ tirées sans collision, "
                 "$Q_{ij}=c_6/\\|\\mathbf{x}^{\\mathrm{ref}}_i-\\mathbf{x}^{\\mathrm{ref}}_j\\|^6$ "
                 "recalculée exactement ; moyennes sur 10 instances. "
                 "Stratégies : MDS (exact), sng.\\ (single), ms.\\ (multistart).")
    lines.append("")
    colspec = (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{1.7cm}"
        "c"
        "r"
        "r"
        "c"
        "@{}"
    )
    for n in sorted({k[0] for k in grouped}):
        for lt in ["L_fixe", "L_adaptatif", "L_optimal"]:
            block = grouped.get((n, lt))
            if not block:
                continue
            best = best_exp2_row(block)
            lines.append(f"\\subsection{{$n={n}$, {L_LABEL[lt]}}}")
            lines.append("{\\footnotesize")
            lines.append("\\setlength{\\tabcolsep}{3pt}")
            lines.append(f"\\begin{{tabular}}{{{colspec}}}")
            lines.append("  \\toprule")
            lines.append("  Formulation & Strat. & $\\mathcal{E}_{\\mathrm{Ryd}}$ & RMSE & Temps \\\\")
            lines.append("  \\midrule")
            order = {"exact": 0, "single": 1, "multistart": 2}
            for r in sorted(
                block,
                key=lambda x: (
                    FORM_ORDER.index(x["formulation"])
                    if x["formulation"] in FORM_ORDER else 99,
                    order.get(x["strategy"], 9),
                ),
            ):
                bold = (
                    math.isfinite(r["obj_rydberg"])
                    and math.isfinite(best["obj_rydberg"])
                    and r["obj_rydberg"] == best["obj_rydberg"]
                    and r["formulation"] == best["formulation"]
                    and r["strategy"] == best["strategy"]
                )
                lines.append(
                    f"  {form_label(r['formulation'])} & "
                    f"{STRATEGY_SHORT.get(r['strategy'], r['strategy'])} & "
                    f"{fmt_sci(r['obj_rydberg'], bold=bold)} & "
                    f"{fmt_rmse(r['rmse_dist'])} & {fmt_time(r['time_s'])} \\\\"
                )
            lines.append("  \\bottomrule")
            lines.append("\\end{tabular}")
            lines.append("}")
            lines.append("")
    return "\n".join(lines) + "\n"


def generate_exp2_instances(log_path):
    rows = parse_exp2_instances(log_path)
    if not rows:
        return f"% No Exp.~2 per-instance log found at {log_path}\n"
    lines = []
    lines.append("% Auto-generated — Exp. 2 per-instance results (Q exacte)")
    lines.append("\\section{Résultats bruts par instance --- Expérience~2}")
    lines.append("\\label{app:exp2-instances}")
    lines.append("")
    lines.append("Source : \\texttt{exp2\\_benchmark.log}. "
                 "Colonnes compactes pour tenir en page.")
    lines.append("")
    colspec = (
        "@{}"
        ">{\\raggedright\\arraybackslash}p{1.55cm}"
        "c"
        "r"
        "r"
        "c"
        ">{\\raggedright\\arraybackslash}p{1.2cm}"
        "@{}"
    )
    cur_inst = None
    cur_key = None
    for r in rows:
        key = (r["instance"], r["n"], r["L_type"], r["L_val"])
        if r["instance"] != cur_inst:
            if cur_inst is not None:
                lines.extend(["  \\bottomrule", "\\end{longtable}", "}"])
            cur_inst = r["instance"]
            lines.append(f"\\subsection{{Instance {cur_inst}/10}}")
        if key != cur_key:
            if cur_key is not None and cur_key[0] == cur_inst:
                lines.extend(["  \\bottomrule", "\\end{longtable}", "}"])
            cur_key = key
            lines.append(
                f"\\subsubsection{{$n={r['n']}$, {L_LABEL_SHORT.get(r['L_type'], r['L_type'])} "
                f"($L={r['L_val']:.4g}$)}}"
            )
            lines.append("{\\footnotesize")
            lines.append("\\setlength{\\tabcolsep}{3pt}")
            lines.append("\\renewcommand{\\arraystretch}{1.03}")
            lines.append(f"\\begin{{longtable}}{{{colspec}}}")
            lines.append("  \\toprule")
            lines.append(
                "  Formulation & Strat. & $\\mathcal{E}_{\\mathrm{Ryd}}$ & RMSE & "
                "Temps & Statut \\\\"
            )
            lines.append("  \\midrule")
            lines.append("  \\endhead")
        lines.append(
            f"  {form_label(r['formulation'])} & "
            f"{STRATEGY_SHORT.get(r['strategy'], r['strategy'])} & "
            f"{fmt_sci(r['obj_rydberg'])} & {fmt_rmse(r['rmse_dist'])} & "
            f"{fmt_time(r['time_s'])} & {fmt_status(r['status'], short=True)} \\\\"
        )
    lines.extend(["  \\bottomrule", "\\end{longtable}", "}"])
    return "\n".join(lines) + "\n"


def parse_exp1_log(log_path):
    """Parse benchmark log output into rows for per-instance appendix."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    rows = []
    inst = None
    n = None
    lt = None
    lval = None
    pat_block = re.compile(r"n=(\d+)\s+\|\s+(L_\w+)\s+\(L=([\d.]+)\)")
    pat_row = re.compile(
        r"\[(\w+)\s*\]\s+err_bruit=([+\-]?[\d.eE+\-infINF]+)\s+t=([\d.]+)s(?:\s+\[([^\]]+)\])?"
    )
    for line in text.splitlines():
        m = re.search(r"Instance (\d+) / (\d+)", line)
        if m:
            inst = int(m.group(1))
            continue
        m = pat_block.search(line)
        if m:
            n, lt, lval = int(m.group(1)), m.group(2), float(m.group(3))
            continue
        m = pat_row.search(line)
        if m and inst is not None and n is not None:
            form = normalize_formulation(m.group(1).strip())
            if not keep_formulation(form):
                continue
            rows.append({
                "instance": inst, "n": n, "L_type": lt, "L_val": lval,
                "formulation": form,
                "err": m.group(2), "time": float(m.group(3)),
                "status": m.group(4) or "",
            })
    return rows


def generate_exp1_instances(log_path):
    rows = parse_exp1_log(log_path)
    if not rows:
        return "% No per-instance log found at " + str(log_path) + "\n"
    lines = []
    lines.append("\\section{Résultats bruts par instance --- Expérience~1}")
    lines.append("\\label{app:exp1-instances}")
    cur_inst = None
    cur_key = None
    for r in rows:
        key = (r["instance"], r["n"], r["L_type"], r["L_val"])
        if r["instance"] != cur_inst:
            if cur_inst is not None:
                lines.extend(["  \\bottomrule", "\\end{longtable}", ""])
            cur_inst = r["instance"]
            lines.append(f"\\subsection{{Instance {cur_inst}/10}}")
        if key != cur_key:
            if cur_key is not None and cur_key[0] == cur_inst:
                lines.extend(["  \\bottomrule", "\\end{longtable}", ""])
            cur_key = key
            lines.append(f"\\subsubsection{{$n={r['n']}$, {L_LABEL.get(r['L_type'], r['L_type'])} ($L={r['L_val']:.4g}$)}}")
            lines.append("\\begin{longtable}{@{}lrrl@{}}")
            lines.append("  \\toprule")
            lines.append("  Formulation & $\\mathcal{E}_{\\mathrm{Ryd}}$ & Temps & Statut \\\\")
            lines.append("  \\midrule")
            lines.append("  \\endhead")
        e = parse_err(r["err"])
        lines.append(
            f"  {FORM_LABEL.get(r['formulation'], r['formulation'])} & "
            f"{fmt_sci(e)} & {fmt_time(r['time'])} & {fmt_status(r['status'])} \\\\"
        )
    lines.extend(["  \\bottomrule", "\\end{longtable}", ""])
    return "\n".join(lines) + "\n"


def main():
    out = ARTICLE / "tables"
    out.mkdir(exist_ok=True)
    (out / "exp1_main.tex").write_text(generate_exp1_main(), encoding="utf-8")
    (out / "exp1_appendix_summary.tex").write_text(generate_exp1_appendix(), encoding="utf-8")

    pasqal = load_pasqal()
    (out / "exp3_main.tex").write_text(generate_exp3_main(pasqal), encoding="utf-8")
    (out / "exp3_appendix.tex").write_text(generate_exp3_appendix(pasqal), encoding="utf-8")

    if EXP2_LOG.exists():
        exp2_summary = parse_exp2_summary(EXP2_LOG)
        (out / "exp2_main.tex").write_text(generate_exp2_main(exp2_summary), encoding="utf-8")
        (out / "exp2_appendix_summary.tex").write_text(
            generate_exp2_appendix(exp2_summary), encoding="utf-8"
        )
        (out / "exp2_appendix_instances.tex").write_text(
            generate_exp2_instances(EXP2_LOG), encoding="utf-8"
        )
        print(f"Generated Exp.~2 tables from {EXP2_LOG}")
    else:
        print(f"Note: {EXP2_LOG} not found — Exp.~2 tables skipped")

    log_path = ARTICLE / "exp1_benchmark.log"
    if log_path.exists():
        (out / "exp1_appendix_instances.tex").write_text(
            generate_exp1_instances(log_path), encoding="utf-8"
        )
        print(f"Generated per-instance appendix from {log_path}")
    else:
        print(f"Note: {log_path} not found — per-instance appendix skipped")

    print(f"Generated tables in {out}/")


if __name__ == "__main__":
    main()
