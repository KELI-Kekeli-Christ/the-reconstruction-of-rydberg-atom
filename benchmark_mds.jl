using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV
using Plots, StatsPlots

Random.seed!(42)
const MOI = JuMP.MOI

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n_values = [2, 3, 20]
dim      = 2
c6       = 1.0
d_min    = 1e-4
z_min    = d_min^2

function nb_restarts_bench(n)
    n <= 3  ? 5 :
    n <= 10 ? 5 :
    n <= 20 ? 3 : 2
end

# ═══════════════════════════════════════════════════════════════
#  §2  UTILITAIRES
# ═══════════════════════════════════════════════════════════════
function generer_init(n, dim, L; seuil=1e-8)
    x = zeros(n, dim)
    for i in 1:n
        x[i, :] = L * rand(dim)
        # On teste le point i contre tous les points précédents (1 à i-1)
        j = 1
        while j < i
            if norm(x[i, :] - x[j, :]) < seuil
                # Collision détectée : on régénère le point i
                x[i, :] = L * rand(dim)
                # On redémarre la vérification pour i à partir de j=1
                j = 1
            else
                j += 1
            end
        end
    end
    return x
end

function erreur_rydberg_vraie(x_sol, Q, c6, n, dim)

    try
        val = 0.0
        for i in 1:n, j in i+1:n
            d = norm(x_sol[i,:] - x_sol[j,:])
            val += (c6 / d^6 - Q[i,j])^2
        end
        return val
    catch
        return Inf
    end
end

# ═══════════════════════════════════════════════════════════════
#  §3  FORMULATION 1 — STANDARD
#  Positions libres dans R² (invariance par translation/rotation)
# ═══════════════════════════════════════════════════════════════
function solve_standard(n, dim, Q, c6, L, x_init)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)

    @variable(m, x[1:n, 1:dim])          
    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end

    @expression(m, d2[i=1:n, j=i+1:n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim)
    )
    for i in 1:n, j in i+1:n
        @constraint(m, d2[i, j] >= z_min)   # contrainte physique
    end
    @objective(m, Min,
        sum((c6 / d2[i, j]^3 - Q[i, j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end

# ═══════════════════════════════════════════════════════════════
#  §4  FORMULATION 2 — REFORMULÉE
# ═══════════════════════════════════════════════════════════════
function solve_reformulee(n, dim, Q, c6, L, x_init)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)

    @variable(m, x2[1:n, 1:dim])         


    @variable(m, z[i=1:n, j=(i+1):n])
    @variable(m, u[i=1:n, j=(i+1):n])
    @variable(m, w[i=1:n, j=(i+1):n])
    @variable(m, r[i=1:n, j=(i+1):n])

    for i in 1:n, d in 1:dim
        set_start_value(x2[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        d_sq = sum((x_init[i,k]-x_init[j,k])^2 for k in 1:dim)
        set_start_value(z[i,j], d_sq)
        set_start_value(u[i,j], d_sq^2)
        set_start_value(w[i,j], d_sq^3)
        set_start_value(r[i,j], c6/d_sq^3)
    end
    for i in 1:n, j in i+1:n
        @constraint(m, r[i,j] * w[i,j] == c6)
        @constraint(m, z[i,j] == sum((x2[i,k]-x2[j,k])^2 for k in 1:dim))
        @constraint(m, w[i,j] == u[i,j] * z[i,j])
        @constraint(m, u[i,j] == z[i,j]^2)
    end
    @objective(m, Min,
        sum((r[i,j]-Q[i,j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x2))
    end
end

# ═══════════════════════════════════════════════════════════════
#  §5  FORMULATION 3 — POLAIRE
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           10000)

    @variable(model, x[1:n, 1:dim])     
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)
    @variable(model, R[i=1:n, j=i+1:n] >= d_min)

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end
    for i in 1:n, j in i+1:n
        dx = x_init[i,1] - x_init[j,1]
        dy = x_init[i,2] - x_init[j,2]
        set_start_value(theta[i,j], atan(dy,dx))
        set_start_value(R[i,j], norm([dx,dy]))
    end

    @variable(model, error_pair[i=1:n, j=i+1:n])
    for i in 1:n, j in i+1:n
        @constraint(model, x[i,1]-x[j,1] == R[i,j]*cos(theta[i,j]))
        @constraint(model, x[i,2]-x[j,2] == R[i,j]*sin(theta[i,j]))
        @constraint(model, error_pair[i,j] == (c6/R[i,j]^6) - Q[i,j])
    end
    @objective(model, Min,
        sum(error_pair[i,j]^2 for i in 1:n for j in i+1:n)
    )

    optimize!(model)
    status = termination_status(model)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(model), value.(x))
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §6  FORMULATION 4 — MDS EXACT (solution algébrique globale)
#
#  Q_ij = c6/d_ij^6  →  d_ij = (c6/Q_ij)^(1/6)  exactement connu
#  → Positionnement par distances via double centrage + eigendecomposition
#  → Erreur Rydberg ≈ 0 garanti si Q parfait, O(n³), sans itération
# ═══════════════════════════════════════════════════════════════
function solve_mds(n, dim, Q, c6)
    # Étape 1 : distances cibles
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        d_ij = (c6 / Q[i,j])^(1/6)
        D[i,j] = d_ij;  D[j,i] = d_ij
    end

    # Étape 2 : double centrage → matrice de Gram
    D2 = D .^ 2
    H  = I - ones(n,n) / n
    B  = Symmetric(-0.5 * H * D2 * H)

    # Étape 3 : décomposition spectrale
    F      = eigen(B)
    λ      = F.values
    V      = F.vectors
    idx    = sortperm(λ, rev=true)[1:dim]
    λ_top  = max.(λ[idx], 0.0)

    # Étape 4 : positions X = V · √Λ
    X_mds = V[:, idx] * Diagonal(sqrt.(λ_top))

    # Translation pour centrer dans R² (pas d'impact sur erreur)
    for d in 1:dim
        X_mds[:, d] .-= minimum(X_mds[:, d])
    end

    return (:MDS_EXACT, X_mds)
end

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
function run_benchmark(n_values, dim, c6)
    df = DataFrame(
        n            = Int[],
        L_type       = String[],
        L_val        = Float64[],
        strategy     = String[],
        formulation  = String[],
        obj_solver   = Float64[],
        obj_rydberg  = Float64[],
        n_restarts   = Int[],
        time_s       = Float64[],
        status       = String[]
    )

    nlp_forms = [
        ("Standard",   solve_standard),
        ("Reformulee", solve_reformulee),
        ("Polaire",    solve_polaire)
    ]

    for n in n_values
        L_configs = [
            ("L_fixe",      10.0),
            ("L_adaptatif", 100.0 * sqrt(n))
        ]

        for (L_type, L) in L_configs
            println("\n" * "═"^62)
            @printf("  n=%-3d  |  %s  (L=%.2f)\n", n, L_type, L)
            println("═"^62)

            x_vrais = generer_init(n, dim, L)
            Q_local = zeros(n, n)
            for i in 1:n, j in i+1:n
                d = norm(x_vrais[i,:]-x_vrais[j,:])
                Q_local[i,j] = Q_local[j,i] = c6/d^6
            end

            n_rs           = nb_restarts_bench(n)
            x_inits_shared = [generer_init(n, dim, L) for _ in 1:n_rs]

            # ── MDS : un seul appel algébrique ─────────────────
            t_mds = @elapsed begin
                _, x_mds = solve_mds(n, dim, Q_local, c6)
            end
            ry_mds = erreur_rydberg_vraie(x_mds, Q_local, c6, n, dim)
            @printf("    [%-12s] obj_rydberg=%+.3e  t=%8.5fs  [EXACT]\n",
                    "MDS_Exact", ry_mds, t_mds)
            push!(df, (n, L_type, L, "exact", "MDS_Exact",
                       0.0, ry_mds, 1, t_mds, "MDS_EXACT"))

            # ── NLP : single et multistart ──────────────────────
            for (strategy_name, use_multistart) in [("single",     false),
                                                     ("multistart", true)]
                println("\n  ── Stratégie : $strategy_name ──")
                sols_config = Dict{String, Union{Nothing, Matrix{Float64}}}()

                for (form_name, solver_fn) in nlp_forms
                    obj_s, obj_ry, stat_s, x_sol, n_used, t_s =run_formulation(solver_fn, n, dim, Q_local, c6, L, x_inits_shared, use_multistart)
                    sols_config[form_name] = x_sol
                    @printf("    [%-12s] obj_solver=%+.3e  obj_rydberg=%+.3e  t=%6.2fs  [%s]\n",
                            form_name, obj_s, obj_ry, t_s,
                            stat_s[1:min(20, length(stat_s))])
                    push!(df, (n, L_type, L, strategy_name, form_name,
                               obj_s, obj_ry, n_used, t_s, stat_s))
                end

                println("\n    ┌── Injection croisée ──────────────────────────┐")
                scores = Float64[]
                all_forms = vcat(nlp_forms, [("MDS_Exact", nothing)])
                sols_config["MDS_Exact"] = x_mds
                for (f, _) in all_forms
                    ry = erreur_rydberg_vraie(sols_config[f], Q_local, c6, n, dim)
                    push!(scores, ry)
                    @printf("    │  x_sol[%-12s] → err_Rydberg = %+.3e\n", f, ry)
                end
                best_form = all_forms[argmin(scores)][1]
                println("    │  → GAGNANT : $best_form")
                println("    └──────────────────────────────────────────────────┘")
            end
        end
    end
    return df
end

const bench_results = run_benchmark(n_values, dim, c6)

# ═══════════════════════════════════════════════════════════════
#  §9  TABLEAU FINAL
# ═══════════════════════════════════════════════════════════════
println("\n\n" * "═"^70)
println("  TABLEAU COMPARATIF FINAL — obj_rydberg (métrique universelle)")
println("═"^70)

println("─"^70)
for grp in groupby(bench_results, [:n, :L_type])
    r = grp[1,:]
    @printf("\n  n=%-3d  %s\n", r.n, r.L_type)
    min_ry = minimum(replace(grp.obj_rydberg, Inf => 1e300))
    for row in eachrow(grp)
        ry_cmp = isinf(row.obj_rydberg) ? 1e300 : row.obj_rydberg
        marker = ry_cmp == min_ry ? " ← ★" : ""
        @printf("    %-12s [%-11s]  obj_ry=%+.4e  t=%7.4fs%s\n",
                row.formulation, row.strategy,
                row.obj_rydberg, row.time_s, marker)
    end
end

csv_path = joinpath(@__DIR__, "benchmark_formulations_test_L_huge.csv")
CSV.write(csv_path, bench_results)
println("\n\nRésultats sauvegardés : $csv_path")
println("═"^70)

# ═══════════════════════════════════════════════════════════════
#  §10  VISUALISATION — 4 GRAPHIQUES (3 formulations NLP + MDS)
# ═══════════════════════════════════════════════════════════════
println("\nGénération des graphiques...")

COULEURS = Dict(
    "Standard"   => :steelblue,
    "Reformulee" => :darkorange,
    "Polaire"    => :seagreen,
    "MDS_Exact"  => :crimson
)
FORMS_NLP = ["Standard", "Reformulee", "Polaire"]
FORMS_ALL = ["Standard", "Reformulee", "Polaire", "MDS_Exact"]
LABELS_NLP = ["Standard", "Reformulée", "Polaire"]
LABELS_ALL = ["Standard", "Reformulée", "Polaire", "MDS Exact"]
n_str = string.(n_values)

safe(x) = isinf(x) || isnan(x) ? NaN : x

function positive_finite_values(series_list)
    vals = Float64[]
    for series in series_list
        append!(vals, filter(x -> isfinite(x) && x > 0, series))
    end
    return vals
end

# Fonction utilitaire de recherche dans le DataFrame
function get_ry(df, n, L_type, strategy, form)
    rows = filter(r -> r.n == n && r.L_type == L_type &&
                       r.strategy == strategy && r.formulation == form,
                  eachrow(df))
    isempty(rows) ? NaN : safe(first(rows).obj_rydberg)
end

function get_time(df, n, L_type, strategy, form)
    rows = filter(r -> r.n == n && r.L_type == L_type &&
                       r.strategy == strategy && r.formulation == form,
                  eachrow(df))
    isempty(rows) ? NaN : first(rows).time_s
end

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
          label=lab, color=COULEURS[form],
          marker=:circle, markersize=6, linewidth=2)
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
          label=lab, color=COULEURS[form],
          marker=:square, markersize=6, linewidth=2)
end
plot!(p2, 1:length(n_values), vals_mds2,
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
    title     = "Temps total — Multistart + MDS, L fixe",
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
          label=lab, color=COULEURS[form],
          marker=:diamond, markersize=7, linewidth=2)
end

# Note MDS sur le graphique
annotate!(p4, length(n_values)/2, 1e4,
          text("MDS toujours ≈ 0\n(pas de notion single/multi)", 8, :gray))

# ─── Assemblage final ────────────────────────────────────────────────────────
fig = plot(p1, p2, p3, p4,
           layout     = (2, 2),
           size       = (1300, 900),
           margin     = 6Plots.mm,
           plot_title = "Benchmark — 3 formulations NLP + MDS exact")

fig_path = joinpath(@__DIR__, "benchmark_visualisation_mds_test_L_huge.png")
savefig(fig, fig_path)
println("\nGraphique sauvegardé : $fig_path")
display(fig)