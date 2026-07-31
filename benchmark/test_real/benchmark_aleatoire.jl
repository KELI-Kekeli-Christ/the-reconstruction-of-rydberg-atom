using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV, Statistics
using Plots, StatsPlots
include("utils.jl")
include("formulations.jl")

#Random.seed!(42)
const MOI = JuMP.MOI

#pour lancer le programme avec du bruit sur les distances faire  BRUIT_DIST_ONLY=1 julia benchmark_bruit.jl
# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
const _bruit_only = get(ENV, "BRUIT_DIST_ONLY", "0") == "1"
const N_INSTANCES = 10
n_values = _bruit_only ? [4,10,20,30] : [4,10,20,30] # preciser la valeur des instances à tester avec bruit dans la premiere liste sans bruit dans la deuxieme 
dim      = 2
c6       = 1.0
d_min    = 0.2
z_min    = d_min^2

function nb_restarts_bench(n)
    n <= 3  ? 5 :
    n <= 10 ? 5 :
    n <= 20 ? 3 : 2
end

# ═══════════════════════════════════════════════════════════════
#  §2  UTILITAIRES
# 

const L_SEED = 10.0
const ALL_L_TYPES = ["L_fixe", "L_adaptatif", "L_optimal"]



# ═══════════════════════════════════════════════════════════════
#  §2b  BRUIT MULTIPLICATIF
#  Seul ajout au code original : fonction pour bruiter Q
# ═══════════════════════════════════════════════════════════════


# Formulations


# ═══════════════════════════════════════════════════════════════
#  §7  RUNNER (NLP avec multistart)
# ═══════════════════════════════════════════════════════════════
function run_formulation(solver_fn, n, dim, Q, c6, L, x_inits_shared,multistart::Bool)
    t0          = time()
    best_obj    = Inf
    best_ry     = Inf
    best_status = "ÉCHEC"
    best_x      = nothing
    n_runs      = multistart ? length(x_inits_shared) : 1
    n_used      = 0

    for k in 1:n_runs
        res = solver_fn(n, dim, Q, c6, L, x_inits_shared[k])
        n_used += 1
        if res !== nothing
            stat, obj, x_sol = res
            ry = erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
            if ry < best_ry
                best_ry     = ry
                best_obj    = obj
                best_status = string(stat)
                best_x      = x_sol
            end
            best_ry < 1e-12 && break
        end
    end

    return (best_obj, best_ry, best_status, best_x, n_used, time()-t0)
end

# ═══════════════════════════════════════════════════════════════
#  §8  BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════════


COULEURS = Dict(
    "Standard"      => :steelblue,
    # "new_standard"  => :darkcyan,
    # "Reformulee"    => :darkorange,
    # "Polaire"       => :seagreen,
    # "Surogate"      => :mediumpurple,
    # "Quadratique"   => :goldenrod,
    "Taylor"        => :chocolate,
    "Taylor_square" => :tomato,
    "Simple_square" => :teal,
    "Simple"        => :orchid,
    "MDS_Exact"     => :crimson
)

color_of(form::AbstractString) = get(COULEURS, form, get(COULEURS, normalize_formulation_name(form), :gray))

display_formulation_name(form::AbstractString) = form

FORMS_NLP = ["Standard", "Taylor", "Taylor_square", "Simple_square", "Simple"]
FORMS_ALL = vcat(FORMS_NLP, ["MDS_Exact"])
LABELS_NLP = ["Standard", "Taylor", "Taylor²", "Simple²", "Simple"]
LABELS_ALL = vcat(LABELS_NLP, ["MDS Exact"])
LINE_STYLES_NLP = [:solid, :dash, :dot, :dashdot, :dashdotdot]

safe(x) = isinf(x) || isnan(x) ? NaN : x



if !_bruit_only
const bench_results, bench_results_raw = run_benchmark(n_values, dim, c6)

# ═══════════════════════════════════════════════════════════════
#  §9  TABLEAU FINAL (moyennes sur 10 instances)
# ═══════════════════════════════════════════════════════════════
println("\n\n" * "═"^70)
println("  TABLEAU COMPARATIF FINAL — moyennes sur 10 instances")
println("  obj_rydberg (métrique universelle) · temps moyen par n")
println("═"^70)

println("─"^70)
for grp in groupby(bench_results, [:n, :L_type])
    r = grp[1,:]
    @printf("\n  n=%-3d  %s\n", r.n, r.L_type)
    min_ry = minimum(replace(grp.obj_rydberg, Inf => 1e300))
    for row in eachrow(grp)
        ry_cmp = isinf(row.obj_rydberg) ? 1e300 : row.obj_rydberg
        marker = ry_cmp == min_ry ? " ← ★" : ""
        @printf("    %-12s [%-11s]  obj_ry=%+.4e  rmse=%.4e  t_moy=%7.4fs%s\n",
                display_formulation_name(row.formulation), row.strategy,
                row.obj_rydberg, row.rmse_dist, row.time_s, marker)
    end
end

csv_path = joinpath(@__DIR__, "benchmark_formulations_test_L_huge.csv")
csv_raw_path = joinpath(@__DIR__, "benchmark_formulations_raw.csv")
CSV.write(csv_path, bench_results)
CSV.write(csv_raw_path, bench_results_raw)
println("\n\nRésultats moyens sauvegardés : $csv_path")
println("Résultats bruts (10 instances) : $csv_raw_path")
println("═"^70)

# ═══════════════════════════════════════════════════════════════
#  §10  VISUALISATION — 4 GRAPHIQUES (3 formulations NLP + MDS)
# ═══════════════════════════════════════════════════════════════
println("\nGénération des graphiques...")

n_str = string.(n_values)

# ─── Plot 1 : Erreur Rydberg — multistart, L fixe ───────────────────────────
series_p1 = [
    [get_ry(bench_results, n, "L_fixe", "multistart", form) for n in n_values]
    for form in FORMS_NLP
]
vals_mds = [get_ry(bench_results, n, "L_fixe", "exact", "MDS_Exact") for n in n_values]
all_p1 = positive_finite_values(vcat(series_p1, [vals_mds]))

if !isempty(all_p1)
    p1 = plot(title="Erreur Rydberg — Multistart + MDS, L fixe (L=10)",
              xlabel="n (atomes)", ylabel="Erreur Rydberg (log₁₀)",
              legend=:topleft, yscale=:log10,
              ylims=(minimum(all_p1)/10, maximum(all_p1)*10),
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
else
    p1 = plot(title="Erreur Rydberg — Multistart + MDS, L fixe (L=10)",
              xlabel="n (atomes)", ylabel="Erreur Rydberg",
              legend=:topleft,
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
end

for (form, lab) in zip(FORMS_NLP, LABELS_NLP)
    vals = get_ry.(Ref(bench_results), n_values, Ref("L_fixe"), Ref("multistart"), Ref(form))
    plot!(p1, 1:length(n_values), vals,
label=lab, color=color_of(form),
          marker=:circle, markersize=6, linewidth=2,
          linestyle=LINE_STYLES_NLP[findfirst(==(form), FORMS_NLP)])
end

# MDS (stratégie "exact")
plot!(p1, 1:length(n_values), vals_mds,
      label="MDS Exact", color=COULEURS["MDS_Exact"],
      marker=:star5, markersize=9, linewidth=2, linestyle=:dash)

# ─── Plot 2 : Erreur Rydberg — multistart, L adaptatif ──────────────────────
series_p2 = [
    [get_ry(bench_results, n, "L_adaptatif", "multistart", form) for n in n_values]
    for form in FORMS_NLP
]
vals_mds2 = [get_ry(bench_results, n, "L_adaptatif", "exact", "MDS_Exact") for n in n_values]
all_p2 = positive_finite_values(vcat(series_p2, [vals_mds2]))

if !isempty(all_p2)
    p2 = plot(title="Erreur Rydberg — Multistart + MDS, L adaptatif (L=2√n)",
              xlabel="n (atomes)", ylabel="Erreur Rydberg (log₁₀)",
              legend=:topleft, yscale=:log10,
              ylims=(minimum(all_p2)/10, maximum(all_p2)*10),
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
else
    p2 = plot(title="Erreur Rydberg — Multistart + MDS, L adaptatif (L=2√n)",
              xlabel="n (atomes)", ylabel="Erreur Rydberg",
              legend=:topleft,
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
end

for (form, lab) in zip(FORMS_NLP, LABELS_NLP)
    vals = get_ry.(Ref(bench_results), n_values, Ref("L_adaptatif"), Ref("multistart"), Ref(form))
    plot!(p2, 1:length(n_values), vals,
label=lab, color=color_of(form),
        marker=:square, markersize=6, linewidth=2,
        linestyle=LINE_STYLES_NLP[findfirst(==(form), FORMS_NLP)])
end
plot!(p2, 1:length(n_values), vals_mds2,
      label="MDS Exact", color=COULEURS["MDS_Exact"],
      marker=:star5, markersize=9, linewidth=2, linestyle=:dash)

annotate!(p2, 1.55, maximum(all_p2) / 1.8,
        text("Sous L adaptatif, les courbes sont presque\nsuperposées : les erreurs sont très proches.",
           8, :gray35))

# ─── Plot 2b : Erreur Rydberg — multistart, L optimal ───────────────────────
series_p2b = [
    [get_ry(bench_results, n, "L_optimal", "multistart", form) for n in n_values]
    for form in FORMS_NLP
]
vals_mds_opt = [get_ry(bench_results, n, "L_optimal", "exact", "MDS_Exact") for n in n_values]
all_p2b = positive_finite_values(vcat(series_p2b, [vals_mds_opt]))

if !isempty(all_p2b)
    p2b = plot(title="Erreur Rydberg — Multistart + MDS, L optimal (L=(c₆/max Q)^(1/6))",
               xlabel="n (atomes)", ylabel="Erreur Rydberg (log₁₀)",
               legend=:topleft, yscale=:log10,
               ylims=(minimum(all_p2b)/10, maximum(all_p2b)*10),
               xticks=(1:length(n_values), n_str),
               yminorgrid=false, size=(600,400))
else
    p2b = plot(title="Erreur Rydberg — Multistart + MDS, L optimal (L=(c₆/max Q)^(1/6))",
               xlabel="n (atomes)", ylabel="Erreur Rydberg",
               legend=:topleft,
               xticks=(1:length(n_values), n_str),
               yminorgrid=false, size=(600,400))
end

for (form, lab) in zip(FORMS_NLP, LABELS_NLP)
    vals = get_ry.(Ref(bench_results), n_values, Ref("L_optimal"), Ref("multistart"), Ref(form))
    plot!(p2b, 1:length(n_values), vals,
label=lab, color=color_of(form),
          marker=:utriangle, markersize=6, linewidth=2,
          linestyle=LINE_STYLES_NLP[findfirst(==(form), FORMS_NLP)])
end
plot!(p2b, 1:length(n_values), vals_mds_opt,
      label="MDS Exact", color=COULEURS["MDS_Exact"],
      marker=:star5, markersize=9, linewidth=2, linestyle=:dash)

# ─── Plot 3 : Temps d'exécution — multistart + MDS, L fixe ─────────────────
# 4 formulations → inner=4
n_f = length(FORMS_ALL)
time_vals = [
    get_time(bench_results, n, "L_fixe",
             form == "MDS_Exact" ? "exact" : "multistart", form)
    for n in n_values for form in FORMS_ALL
]

p3 = groupedbar(
    repeat(n_str, inner=n_f),
    time_vals,
    group     = repeat(LABELS_ALL, outer=length(n_values)),
    title     = "Temps moyen — Multistart + MDS, L fixe (10 instances)",
    xlabel    = "n (atomes)",
    ylabel    = "Temps (secondes)",
    color     = reshape([COULEURS[f] for f in FORMS_ALL], 1, n_f),
    bar_width = 0.7,
    legend    = :topleft,
    size      = (600, 400)
)

# ─── Plot 4 : Gain multistart vs single — L fixe (NLP seulement) ────────────
# MDS n'a pas de notion single/multistart → exclu de ce graphique
ratio_series = Dict{String, Vector{Float64}}()
all_ratios = Float64[]
for form in FORMS_NLP
    ratios = Float64[]
    for n in n_values
        ry_s = get_ry(bench_results, n, "L_fixe", "single", form)
        ry_m = get_ry(bench_results, n, "L_fixe", "multistart", form)
        ratio = if isfinite(ry_s) && isfinite(ry_m) && ry_m > 0
            ry_s / ry_m
        elseif !isfinite(ry_s) && isfinite(ry_m)
            1e6
        else
            NaN
        end
        push!(ratios, ratio)
    end
    ratio_series[form] = ratios
    append!(all_ratios, filter(x -> isfinite(x) && x > 0, ratios))
end

if !isempty(all_ratios)
    p4 = plot(title="Gain multistart (ratio single/multi) — L fixe",
              xlabel="n (atomes)", ylabel="Ratio obj_single / obj_multi",
              legend=:topright, yscale=:log10,
              ylims=(minimum(all_ratios)/10, maximum(all_ratios)*10),
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
else
    p4 = plot(title="Gain multistart (ratio single/multi) — L fixe",
              xlabel="n (atomes)", ylabel="Ratio obj_single / obj_multi",
              legend=:topright,
              xticks=(1:length(n_values), n_str),
              yminorgrid=false, size=(600,400))
end

hline!(p4, [1.0], linestyle=:dash, color=:black,
       label="ratio=1 (aucun gain)", linewidth=1)

for (form, lab) in zip(FORMS_NLP, LABELS_NLP)
    ratios = ratio_series[form]
plot!(p4, 1:length(n_values), ratios,
          label=lab, color=color_of(form),
          marker=:diamond, markersize=7, linewidth=2)
end

# Note MDS sur le graphique
annotate!(p4, length(n_values)/2, 1e4,
          text("MDS toujours ≈ 0\n(pas de notion single/multi)", 8, :gray))

# ─── Assemblage final ────────────────────────────────────────────────────────
fig = plot(p1, p2, p2b, p3, p4,
           layout     = (3, 2),
           size       = (1300, 1200),
           margin     = 6Plots.mm,
           plot_title = "Benchmark — 3 formulations NLP + MDS exact")

fig_path = joinpath(@__DIR__, "benchmark_visualisation_mds_test_L_huge.png")
savefig(fig, fig_path)
println("\nGraphique sauvegardé : $fig_path")
display(fig)



fig_rmse = plot(
    plot_rmse_multistart(bench_results, "L_fixe", "L fixe"),
    plot_rmse_multistart(bench_results, "L_adaptatif", "L adaptatif"),
    plot_rmse_multistart(bench_results, "L_optimal", "L optimal"),
    layout=(3, 1), size=(700, 1100), margin=6Plots.mm,
    plot_title="RMSE distances — toutes formulations et L"
)
fig_rmse_path = joinpath(@__DIR__, "benchmark_rmse_dist.png")
savefig(fig_rmse, fig_rmse_path)
println("Graphique RMSE sauvegardé : $fig_rmse_path")
display(fig_rmse)
end  # !_bruit_only

# ═══════════════════════════════════════════════════════════════
#  §11  BENCHMARK AVEC Q_bruit_dist
#  Q généré par distances aléatoires dans [0, L] (pas de x vrai).
#  Métrique principale : err_vs_bruit (ajustement à Q_bruit_dist).
# ═══════════════════════════════════════════════════════════════
println("\n\n" * "═"^70)
println("  BENCHMARK Q_bruit_dist — distances aléatoires dans [0, L]")
println("  Moyennes sur $N_INSTANCES instances")
println("═"^70)

bruit_niveaux = [0.0]

df_bruit = DataFrame(
    instance    = Int[],
    n           = Int[],
    L_type      = String[],
    L_val       = Float64[],
    sigma       = Float64[],
    formulation = String[],
    err_vs_vrai = Float64[],
    err_vs_bruit= Float64[],
    rmse_dist   = Float64[],
    time_s      = Float64[],
    Q           = String[],
    X           = String[],
    x_opt       = String[]
)

mds_positions = Dict{Tuple{Int,String,Float64}, Tuple{Matrix{Float64}, Matrix{Float64}}}()

nlp_forms_bruit = [
    ("Standard",      solve_standard),
    # ("new_standard",  solve_new_standard),
    # ("Reformulee",    solve_reformulee),
    # ("Polaire",       solve_polaire),
    # ("Surogate",      solve_surogate),
    # ("Quadratique",   solve_quadratique),
    ("Taylor",        solve_taylor),
    ("Taylor_square", solve_taylor_square),
    ("Simple_square", solve_simple_square),
    ("Simple",        solve_simple),
]
for inst in 1:N_INSTANCES
    println("\n── Instance $inst / $N_INSTANCES ──")
    for n in n_values
        for L_type in ALL_L_TYPES
            if L_type == "L_optimal"
                Q_temp  = Q_bruit_dist(n, c6, L_SEED)
                L_bruit = L_optimal_from_Q(Q_temp, c6, n)
            else
                L_bruit = L_value(L_type, n; c6=c6)
            end

            println("\n" * "═"^62)
            @printf("  n=%-3d  |  %s  (L=%.4f)\n", n, L_type, L_bruit)
            println("═"^62)

            for theta_bruit in bruit_niveaux
                @printf("\n  ── Q_bruit_dist ──\n")

                Q_bruit = Q_bruit_dist(n, c6, L_bruit)
                if L_type == "L_optimal"
                    L_bruit = L_optimal_from_Q(Q_bruit, c6, n)
                end
                Q_str = matrix_to_csv_str(Q_bruit)
                X_str = ""

                n_rs           = nb_restarts_bench(n)
                x_inits_shared = [generer_init(n, dim, L_bruit) for _ in 1:n_rs]

                # MDS sur Q_bruit_dist
                t_mds = @elapsed begin
                    _, x_mds = solve_mds(n, dim, Q_bruit, c6)
                end
                eb_mds = erreur_rydberg_vraie(x_mds, Q_bruit, c6, n, dim)
                @printf("    [%-12s] err_bruit=%+.3e  t=%.4fs\n",
                        "MDS_Exact", eb_mds, t_mds)
                x_opt_str = matrix_to_csv_str(x_mds)
                push!(df_bruit, (inst, n, L_type, L_bruit, theta_bruit, "MDS_Exact",
                                 NaN, eb_mds, NaN, t_mds, Q_str, X_str, x_opt_str))

                # NLP multistart sur Q_bruit_dist
                for (form_name, solver_fn) in nlp_forms_bruit
                    ry_bruit, stat_s, x_sol, t_s =
                        begin
                            t0 = time()
                            br = Inf; bs = "ÉCHEC"; bx = nothing
                            for k in 1:n_rs
                                res = solver_fn(n, dim, Q_bruit, c6, L_bruit, x_inits_shared[k])
                                if res !== nothing
                                    stat, obj, xs = res
                                    ry = erreur_rydberg_vraie(xs, Q_bruit, c6, n, dim)
                                    if ry < br; br = ry; bs = string(stat); bx = xs; end
                                    br < 1e-12 && break
                                end
                            end
                            (br, bs, bx, time()-t0)
                        end
                    @printf("    [%-12s] err_bruit=%+.3e  t=%.2fs  [%s]\n",
                            display_formulation_name(form_name), ry_bruit, t_s,
                            stat_s[1:min(20, length(stat_s))])
                    push!(df_bruit, (inst, n, L_type, L_bruit, theta_bruit, form_name,
                                     NaN, ry_bruit, NaN, t_s, Q_str, X_str, ""))
                end
            end
        end
    end
end

df_bruit_mean = aggregate_bruit_benchmark(df_bruit)
print_bruit_means(df_bruit_mean, N_INSTANCES)

csv_bruit_raw = joinpath(@__DIR__, "benchmark_bruit_raw.csv")
csv_bruit = joinpath(@__DIR__, "benchmark_bruit_test.csv")
CSV.write(csv_bruit_raw, df_bruit)
CSV.write(csv_bruit, df_bruit_mean)
println("\n\nRésultats bruit moyens sauvegardés : $csv_bruit")
println("Résultats bruit bruts ($N_INSTANCES instances) : $csv_bruit_raw")

# ═══════════════════════════════════════════════════════════════
#  §12  VISUALISATION DU BRUIT
#  Deux métriques : err_vs_vrai et err_vs_bruit
#  Chaque figure compare L_fixe, L_adaptatif et L_optimal.
# ═══════════════════════════════════════════════════════════════
println("\nGénération des graphiques de bruit...")



fig_bruit_vrai = plot_bruit_grid(df_bruit_mean, :err_vs_vrai, "Erreur vs Q vrai (moyenne $N_INSTANCES inst.)")
fig_bruit_bruit = plot_bruit_grid(df_bruit_mean, :err_vs_bruit, "Erreur vs Q bruité (moyenne $N_INSTANCES inst.)")
fig_bruit_rmse = plot_bruit_grid(df_bruit_mean, :rmse_dist, "RMSE distances (moyenne $N_INSTANCES inst.)")

path_bruit_vrai = joinpath(@__DIR__, "benchmark_bruit_vs_vrai.png")
path_bruit_bruit = joinpath(@__DIR__, "benchmark_bruit_vs_bruit.png")
path_bruit_rmse = joinpath(@__DIR__, "benchmark_bruit_rmse.png")

savefig(fig_bruit_vrai, path_bruit_vrai)
savefig(fig_bruit_bruit, path_bruit_bruit)
savefig(fig_bruit_rmse, path_bruit_rmse)

println("Graphique bruit sauvegardé : $path_bruit_vrai")
println("Graphique bruit sauvegardé : $path_bruit_bruit")
println("Graphique bruit sauvegardé : $path_bruit_rmse")
display(fig_bruit_vrai)
display(fig_bruit_bruit)
display(fig_bruit_rmse)

# ═══════════════════════════════════════════════════════════════
#  §13  VISUALISATION DES POSITIONS — MDS Exact vs X vrai
#  Un sous-graphe par (n, L_type) pour toutes les valeurs de n
# ═══════════════════════════════════════════════════════════════
println("\nGénération des graphiques MDS Exact (X vrai vs x_opt)...")


    #n_sub = length(subplots)
    #n_cols = min(3, n_sub)
    #n_rows = cld(n_sub, n_cols)

    #return plot(subplots...,
               # layout=(n_rows, n_cols),
                #size=(520 * n_cols, 420 * n_rows),
                #margin=6Plots.mm,
                #plot_title="MDS Exact — positions X vrai vs x_opt (σ=$(sigma))")
#end

sigma_plot = first(bruit_niveaux)
if !isempty(mds_positions)
    fig_mds_positions = plot_mds_positions_grid(mds_positions, sigma_plot)

    path_mds_positions = joinpath(@__DIR__, "benchmark_positions_mds_exact.png")
    savefig(fig_mds_positions, path_mds_positions)
    println("Graphique positions MDS Exact sauvegardé : $path_mds_positions")
    display(fig_mds_positions)
end

# ═══════════════════════════════════════════════════════════════
#  §14  FONCTION START() — LANCEUR CONFIGURÉ
# ═══════════════════════════════════════════════════════════════
function start(;
    n_vals::Vector{Int} = [3, 10],
    formulations::Vector{String} = ["Standard", "Taylor", "Taylor_square", "Simple_square", "Simple", "MDS"],
    L_types::Vector{String} = ALL_L_TYPES,
    noise_levels::Vector{Float64} = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30],
    vis_n::Int = maximum(n_vals),
    vis_L::String = "L_adaptatif",
    vis_sigma::Float64 = 0.10,
    verbose::Bool = true
)
    """
    Lance le benchmark complet avec configuration flexible.

    # Paramètres
    - n_vals: tailles de problèmes à tester
    - formulations: liste de formulations ("Standard", "Reformulee", "Polaire", "Surogate", "Quadratique", "MDS")
    - L_types: configurations de L ("L_fixe", "L_adaptatif", "L_optimal")
    - noise_levels: niveaux de bruit à tester
    - vis_n: taille pour visualisation des positions
    - vis_L: type de L pour visualisation
    - vis_sigma: niveau de bruit pour visualisation
    - verbose: affiche les détails

    # Exemples
    start()  # tous les defaults
    start(n_vals=[2,3], formulations=["MDS", "Polaire"])
    start(formulations=["Standard", "Polaire"])
    """

    canonical_forms = unique(normalize_formulation_name.(formulations))
    canonical_forms = filter(f -> f in ["Standard", "Taylor", "Taylor_square", "Simple_square", "Simple", "MDS_Exact"], canonical_forms)

    verbose && println("\n" * "═"^70)
    verbose && println("  BENCHMARK CONFIGURÉ")
    verbose && println("═"^70)
    verbose && @printf("  n: %s\n", n_vals)
    verbose && @printf("  Formulations: %s\n", join(display_formulation_name.(canonical_forms), ", "))
    verbose && @printf("  L: %s\n", join(L_types, ", "))
    verbose && @printf("  Niveaux de bruit: %s\n", join(string.(noise_levels), ", "))
    verbose && println("═"^70 * "\n")

    if isempty(canonical_forms)
        error("Aucune formulation valide. Utiliser Standard/std, Taylor, Taylor_square, Simple_square, Simple, MDS/mds.")
    end

    df = run_benchmark(n_vals, dim, c6; formulations=canonical_forms, L_types=L_types)
    
    if verbose
        @printf("\n  Résultats calculés: %d lignes\n", nrow(df))
        for grp in groupby(df, [:formulation])
            f = first(grp).formulation
            @printf("  - %s : %d lignes\n", f, nrow(grp))
        end
    end

    return df
end

# ─── LANCER LE BENCHMARK ───────────────────────────────────────
if abspath(PROGRAM_FILE) == @__FILE__
    if !_bruit_only
        #start(n_vals=[1000], formulations=["mds"], L_types=ALL_L_TYPES, noise_levels=[0])
    end
end

