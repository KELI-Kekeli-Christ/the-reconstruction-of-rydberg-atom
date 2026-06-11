using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV,Pkg
Pkg.add("StatsPlots")
using Plots, StatsPlots

Random.seed!(42)
const MOI = JuMP.MOI

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n_values = [2, 3, 10, 20]
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
function generer_init(n, dim, L; seuil=1e-3)
    x = L * rand(n, dim)
    for i in 1:n
        while norm(x[i,:]) < seuil
            x[i,:] = L * rand(dim)
        end
        for j in 1:i-1
            t = 0
            while norm(x[i,:] - x[j,:]) < seuil
                x[i,:] .+= seuil * rand(dim)
                x[i,:]   = clamp.(x[i,:], 0.0, L)
                t += 1
                if t > 100; x[i,:] = L * rand(dim); t = 0; end
            end
        end
    end
    return x
end

function erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
    isnothing(x_sol) && return Inf
    try
        val = 0.0
        for i in 1:n, j in i+1:n
            d = max(norm(x_sol[i,:] - x_sol[j,:]), 1e-5)
            val += (c6 / d^6 - Q[i,j])^2
        end
        return val
    catch
        return Inf
    end
end

# ═══════════════════════════════════════════════════════════════
#  §3  FORMULATION 1 — STANDARD
# ═══════════════════════════════════════════════════════════════
function solve_standard(n, dim, Q, c6, L, x_init)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           3000)
    set_optimizer_attribute(m, "bound_relax_factor", 1e-8)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end

    @expression(m, d2[i=1:n, j=i+1:n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim)
    )
    for i in 1:n, j in i+1:n
        @constraint(m, d2[i, j] >= z_min)
    end
    @objective(m, Min,
        sum((c6 / d2[i, j]^3 - Q[i, j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §4  FORMULATION 2 — REFORMULÉE
# ═══════════════════════════════════════════════════════════════
function solve_reformulee(n, dim, Q, c6, L, x_init)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           3000)
    set_optimizer_attribute(m, "bound_relax_factor", 1e-8)

    @variable(m, 0 <= x2[1:n, 1:dim] <= L)
    z_max = dim * L^2
    r_min = c6 / z_max^3
    r_max = c6 / z_min^3
    @variable(m, z_min <= z[i=1:n, j=(i+1):n] <= z_max)
    @variable(m, 0.0   <= u[i=1:n, j=(i+1):n] <= z_max^2)
    @variable(m, 0.0   <= w[i=1:n, j=(i+1):n] <= z_max^3)
    @variable(m, r_min <= r[i=1:n, j=(i+1):n] <= r_max)

    for i in 1:n, d in 1:dim
        set_start_value(x2[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        d_sq = max(sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim), z_min)
        set_start_value(z[i, j], d_sq)
        set_start_value(u[i, j], d_sq^2)
        set_start_value(w[i, j], d_sq^3)
        set_start_value(r[i, j], clamp(c6 / d_sq^3, r_min, r_max))
    end
    for i in 1:n, j in i+1:n
        @constraint(m, r[i, j] * w[i, j] == c6)
        @constraint(m, z[i, j] == sum((x2[i,k] - x2[j,k])^2 for k in 1:dim))
        @constraint(m, w[i, j] == u[i, j] * z[i, j])
        @constraint(m, u[i, j] == z[i, j]^2)
    end
    @objective(m, Min,
        sum((r[i, j] - Q[i, j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x2))
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §5  FORMULATION 3 — POLAIRE
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)

    @variable(model, 0 <= x[1:n, 1:dim] <= L)
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)
    @variable(model, R[i=1:n, j=i+1:n] >= d_min)

    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        dx = x_init[i, 1] - x_init[j, 1]
        dy = x_init[i, 2] - x_init[j, 2]
        set_start_value(theta[i, j], atan(dy, dx))
        set_start_value(R[i, j], max(norm([dx, dy]), d_min))
    end

    @variable(model, error_pair[i=1:n, j=i+1:n])
    for i in 1:n, j in i+1:n
        @constraint(model, x[i,1] - x[j,1] == R[i,j] * cos(theta[i,j]))
        @constraint(model, x[i,2] - x[j,2] == R[i,j] * sin(theta[i,j]))
        @constraint(model, error_pair[i,j] == (c6 / R[i,j]^6) - Q[i,j])
    end
    @objective(model, Min,
        sum(error_pair[i,j]^2 for i in 1:n for j in i+1:n)
    )

    optimize!(model)
    status = termination_status(model)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.SLOW_PROGRESS]
        return (status, objective_value(model), value.(x))
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §6  RUNNER
# ═══════════════════════════════════════════════════════════════
function run_formulation(solver_fn, n, dim, Q, c6, L, x_inits_shared,
                         multistart::Bool)
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

    t_exec = time() - t0
    return (best_obj, best_ry, best_status, best_x, n_used, t_exec)
end

# ═══════════════════════════════════════════════════════════════
#  §7  BOUCLE PRINCIPALE
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

    formulations = [
        ("Standard",   solve_standard),
        ("Reformulee", solve_reformulee),
        ("Polaire",    solve_polaire)
    ]

    for n in n_values
        L_configs = [
            ("L_fixe",      10.0),
            ("L_adaptatif", 2.0 * sqrt(n))
        ]

        for (L_type, L) in L_configs
            println("\n" * "═"^60)
            @printf("  n=%-3d  |  %s  (L=%.2f)\n", n, L_type, L)
            println("═"^60)

            x_vrais = generer_init(n, dim, L)
            Q_local = zeros(n, n)
            for i in 1:n, j in i+1:n
                d = max(norm(x_vrais[i,:] - x_vrais[j,:]), 1e-5)
                Q_local[i,j] = Q_local[j,i] = c6 / d^6
            end

            n_rs           = nb_restarts_bench(n)
            x_inits_shared = [generer_init(n, dim, L) for _ in 1:n_rs]

            for (strategy_name, use_multistart) in [("single",     false),
                                                     ("multistart", true)]
                println("\n  ── Stratégie : $strategy_name ──")
                sols_config = Dict{String, Union{Nothing, Matrix{Float64}}}()

                for (form_name, solver_fn) in formulations
                    obj_s, obj_ry, stat_s, x_sol, n_used, t_s =
                        run_formulation(solver_fn, n, dim, Q_local, c6, L,
                                        x_inits_shared, use_multistart)
                    sols_config[form_name] = x_sol
                    @printf("    [%-12s] obj_solver=%+.3e  obj_rydberg=%+.3e  t=%6.2fs  [%s]\n",
                            form_name, obj_s, obj_ry, t_s,
                            stat_s[1:min(20, length(stat_s))])
                    push!(df, (n, L_type, L, strategy_name, form_name,
                               obj_s, obj_ry, n_used, t_s, stat_s))
                end

                println("\n    ┌── Injection croisée ────────────────────────┐")
                scores = Float64[]
                for (f, _) in formulations
                    ry = erreur_rydberg_vraie(sols_config[f], Q_local, c6, n, dim)
                    push!(scores, ry)
                    @printf("    │  x_sol[%-12s] → err_Rydberg = %+.3e\n", f, ry)
                end
                best_form = formulations[argmin(scores)][1]
                println("    │  → GAGNANT : $best_form")
                println("    └────────────────────────────────────────────────┘")
            end
        end
    end
    return df
end

const bench_results = run_benchmark(n_values, dim, c6)

# ═══════════════════════════════════════════════════════════════
#  §8  TABLEAU FINAL
# ═══════════════════════════════════════════════════════════════
println("\n\n" * "═"^70)
println("  TABLEAU COMPARATIF FINAL — obj_rydberg (métrique universelle)")
println("═"^70)

summary = combine(
    groupby(bench_results, [:n, :L_type, :strategy]),
    [:formulation, :obj_rydberg, :time_s] =>
        ((f, o, t) -> begin
            idx = argmin(replace(o, Inf => 1e300))
            (winner=f[idx], best_obj=o[idx], time_winner=t[idx])
        end) => AsTable
)
println(summary)
println()

println("─"^70)
println("  Détail par formulation :")
println("─"^70)
for grp in groupby(bench_results, [:n, :L_type, :strategy])
    r = grp[1, :]
    @printf("\n  n=%-3d  %s  %s\n", r.n, r.L_type, r.strategy)
    min_ry = minimum(replace(grp.obj_rydberg, Inf => 1e300))
    for row in eachrow(grp)
        ry_cmp = isinf(row.obj_rydberg) ? 1e300 : row.obj_rydberg
        marker = ry_cmp == min_ry ? " ← ★ MEILLEUR" : ""
        @printf("    %-12s  obj_ry=%+.4e  t=%6.2fs  n_rs=%-2d%s\n",
                row.formulation, row.obj_rydberg, row.time_s, row.n_restarts, marker)
    end
end

csv_path = joinpath(@__DIR__, "benchmark_formulations.csv")
CSV.write(csv_path, bench_results)
println("\n\nRésultats sauvegardés : $csv_path")
println("═"^70)

# ═══════════════════════════════════════════════════════════════
#  §9  VISUALISATION — 4 GRAPHIQUES
# ═══════════════════════════════════════════════════════════════
println("\nGénération des graphiques...")

# Palette et labels communs
COULEURS   = Dict("Standard" => :steelblue, "Reformulee" => :darkorange, "Polaire" => :seagreen)
FORMS      = ["Standard", "Reformulee", "Polaire"]
LABELS     = ["Standard", "Reformulée", "Polaire"]
n_str      = string.(n_values)

# Remplace Inf par NaN pour que Plots saute les barres manquantes
safe(x) = isinf(x) ? NaN : x

# ─── Graphique 1 : Erreur Rydberg (log) — multistart, L_fixe ────────────────
# Compare la qualité des 3 formulations pour différents n
# Axe log10 pour visualiser les ordres de grandeur
p1 = plot(
    title  = "Erreur Rydberg — Multistart, L fixe (L=10)",
    xlabel = "n (nombre d'atomes)",
    ylabel = "Erreur Rydberg (log₁₀)",
    legend = :topleft,
    yscale = :log10,
    xticks = (1:length(n_values), n_str),
    yminorgrid = true,
    gridcolor = :lightgray,
    size = (600, 400)
)

for (form, lab) in zip(FORMS, LABELS)
    vals = [
        safe(only(filter(r -> r.n == n && r.L_type == "L_fixe" &&
                              r.strategy == "multistart" &&
                              r.formulation == form,
                         eachrow(bench_results)).obj_rydberg))
        for n in n_values
    ]
    plot!(p1, 1:length(n_values), vals,
          label     = lab,
          color     = COULEURS[form],
          marker    = :circle,
          markersize = 6,
          linewidth  = 2)
end

# ─── Graphique 2 : Erreur Rydberg (log) — multistart, L adaptatif ───────────
p2 = plot(
    title  = "Erreur Rydberg — Multistart, L adaptatif (L=2√n)",
    xlabel = "n (nombre d'atomes)",
    ylabel = "Erreur Rydberg (log₁₀)",
    legend = :topleft,
    yscale = :log10,
    xticks = (1:length(n_values), n_str),
    yminorgrid = true,
    gridcolor = :lightgray,
    size = (600, 400)
)

for (form, lab) in zip(FORMS, LABELS)
    vals = [
        safe(only(filter(r -> r.n == n && r.L_type == "L_adaptatif" &&
                              r.strategy == "multistart" &&
                              r.formulation == form,
                         eachrow(bench_results)).obj_rydberg))
        for n in n_values
    ]
    plot!(p2, 1:length(n_values), vals,
          label     = lab,
          color     = COULEURS[form],
          marker    = :square,
          markersize = 6,
          linewidth  = 2)
end

# ─── Graphique 3 : Temps d'exécution — multistart, L fixe ───────────────────
# Barres groupées : une couleur par formulation, un groupe par n
p3 = groupedbar(
    repeat(n_str, inner=3),
    [
        safe(only(filter(r -> r.n == n && r.L_type == "L_fixe" &&
                              r.strategy == "multistart" &&
                              r.formulation == form,
                         eachrow(bench_results)).time_s))
        for n in n_values for form in FORMS
    ],
    group  = repeat(LABELS, outer=length(n_values)),
    title  = "Temps total — Multistart, L fixe",
    xlabel = "n (nombre d'atomes)",
    ylabel = "Temps (secondes)",
    color  = [COULEURS[f] for f in FORMS],
    bar_width = 0.6,
    legend = :topleft,
    gridcolor = :lightgray,
    size = (600, 400)
)

# ─── Graphique 4 : Gain du multistart — L fixe ──────────────────────────────
# Rapport obj_single / obj_multi (> 1 = multistart améliore)
# Montre concrètement à quel point le multistart aide chaque formulation
p4 = plot(
    title  = "Gain du multistart (single/multi) — L fixe",
    xlabel = "n (nombre d'atomes)",
    ylabel = "Ratio obj_single / obj_multistart",
    legend = :topright,
    yscale = :log10,
    xticks = (1:length(n_values), n_str),
    yminorgrid = true,
    gridcolor = :lightgray,
    size = (600, 400)
)

hline!(p4, [1.0], linestyle=:dash, color=:black, label="ratio=1 (aucun gain)", linewidth=1)

for (form, lab) in zip(FORMS, LABELS)
    ratios = Float64[]
    for n in n_values
        rows_s = filter(r -> r.n == n && r.L_type == "L_fixe" &&
                             r.strategy == "single" &&
                             r.formulation == form,
                        eachrow(bench_results))
        rows_m = filter(r -> r.n == n && r.L_type == "L_fixe" &&
                             r.strategy == "multistart" &&
                             r.formulation == form,
                        eachrow(bench_results))
        ry_s = only(rows_s).obj_rydberg
        ry_m = only(rows_m).obj_rydberg

        # Si les deux sont finis → calcul du ratio
        # Si single=Inf et multi fini → ratio symbolique 1e6 (grand gain)
        # Sinon → NaN (formulation échec total)
        ratio = if isfinite(ry_s) && isfinite(ry_m) && ry_m > 0
            ry_s / ry_m
        elseif isinf(ry_s) && isfinite(ry_m)
            1e6
        else
            NaN
        end
        push!(ratios, ratio)
    end
    plot!(p4, 1:length(n_values), ratios,
          label     = lab,
          color     = COULEURS[form],
          marker    = :diamond,
          markersize = 7,
          linewidth  = 2)
end

# ─── Assemblage des 4 graphiques en une figure 2×2 ──────────────────────────
fig = plot(p1, p2, p3, p4,
           layout  = (2, 2),
           size    = (1200, 800),
           margin  = 5Plots.mm,
           plot_title = "Benchmark — Comparaison des 3 formulations Rydberg")

fig_path = joinpath(@__DIR__, "benchmark_visualisation.png")
savefig(fig, fig_path)
println("\nGraphique sauvegardé : $fig_path")

display(fig)