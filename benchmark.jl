using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV

Random.seed!(42)
const MOI = JuMP.MOI

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n_values = [2, 3, 10, 20]
dim      = 2
c6       = 1.0
d_min    = 1e-4
z_min    = d_min^2   # = 1e-8

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


    # Expression d² définie une seule fois → un seul arbre symbolique
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
    # Variables libres — aucune fixation arbitraire

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
#
#  CORRECTION 1 : 0 <= x[1:n, 1:dim] (pas 1:2)
#  → même type de conteneur JuMP que Standard et Reformulée
#
#  CORRECTION BONUS : theta borné dans [-π, π]
#  → réduit l'espace de recherche (angle physique ∈ [-π, π])
#  → Hessien mieux conditionné
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)

    # CORRECTION : 1:dim au lieu de 1:2 → conteneur identique aux autres
    @variable(model, 0 <= x[1:n, 1:dim] <= L)

    # CORRECTION BONUS : theta physiquement borné dans [-π, π]
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)

    @variable(model, R[i=1:n, j=i+1:n] >= d_min)

    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        dx = x_init[i, 1] - x_init[j, 1]
        dy = x_init[i, 2] - x_init[j, 2]
        set_start_value(theta[i, j], atan(dy, dx))    # ∈ [-π, π] par définition
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
#
#  CORRECTION : retour nommé bench_results (pas results)
#  → évite le conflit de portée avec la variable globale résultante
# ═══════════════════════════════════════════════════════════════
function run_benchmark(n_values, dim, c6)

    # Variable locale à la fonction — pas de conflit avec la globale
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

# ═══════════════════════════════════════════════════════════════
#  LANCEMENT — variable globale nommée bench_results
#  → pas de conflit avec df interne à run_benchmark
# ═══════════════════════════════════════════════════════════════
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