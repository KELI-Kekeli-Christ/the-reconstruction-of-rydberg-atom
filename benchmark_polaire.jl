using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV

Random.seed!(42)
const MOI = JuMP.MOI

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n_values = [2, 3, 10]
dim      = 2
c6       = 1.0
d_min    = 1e-8
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
#  §3  FORMULATION POLAIRE
#  Variables : x (positions), R (distances), θ (angles)
#  Contraintes :
#    xi - xj = R[i,j] · (cos θ, sin θ)   (décomposition polaire)
#    error[i,j] = c6/R[i,j]^6 - Q[i,j]
#  Objectif : min Σ_{i<j} error[i,j]²
#
#  Protection singularité : R >= d_min (distance minimale physique)
#  Conditionnement       : θ ∈ [-π, π]  (angle physique borné)
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)

    @variable(model,  x[1:n, 1:dim])
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
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(model), value.(x))
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §4  RUNNER
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
#  §5  BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════════
function run_benchmark(n_values, dim, c6)
    df = DataFrame(
        n           = Int[],
        L_type      = String[],
        L_val       = Float64[],
        strategy    = String[],
        obj_solver  = Float64[],
        obj_rydberg = Float64[],
        n_restarts  = Int[],
        time_s      = Float64[],
        status      = String[]
    )

    for n in n_values
        L_configs = [
            ("L_fixe",      10.0),
            ("L_adaptatif", 2.0 * sqrt(n))
        ]

        for (L_type, L) in L_configs
            println("\n" * "═"^55)
            @printf("  [Polaire]  n=%-3d  |  %s  (L=%.2f)\n", n, L_type, L)
            println("═"^55)

            x_vrais = generer_init(n, dim, L)
            Q_local = zeros(n, n)
            for i in 1:n, j in i+1:n
                d = max(norm(x_vrais[i,:] - x_vrais[j,:]), 1e-5)
                Q_local[i,j] = Q_local[j,i] = c6 / d^6
            end

            n_rs           = nb_restarts_bench(n)
            x_inits_shared = [generer_init(n, dim, L) for _ in 1:n_rs]

            for (strategy_name, use_multistart) in [("single", false),
                                                     ("multistart", true)]
                obj_s, obj_ry, stat_s, x_sol, n_used, t_s =
                    run_formulation(solve_polaire, n, dim, Q_local, c6, L,
                                    x_inits_shared, use_multistart)

                @printf("  [%-11s] obj_solver=%+.3e  obj_rydberg=%+.3e  t=%6.2fs  [%s]\n",
                        strategy_name, obj_s, obj_ry, t_s,
                        stat_s[1:min(20, length(stat_s))])

                push!(df, (n, L_type, L, strategy_name,
                           obj_s, obj_ry, n_used, t_s, stat_s))
            end
        end
    end
    return df
end

const bench_results = run_benchmark(n_values, dim, c6)

println("\n\n" * "═"^55)
println("  RÉSULTATS — FORMULATION POLAIRE")
println("═"^55)
for grp in groupby(bench_results, [:n, :L_type, :strategy])
    r = grp[1,:]
    @printf("  n=%-3d  %s  %s  →  obj_ry=%+.4e  t=%.2fs\n",
            r.n, r.L_type, r.strategy, r.obj_rydberg, r.time_s)
end

csv_path = joinpath(@__DIR__, "results_polaire.csv")
CSV.write(csv_path, bench_results)
println("\nRésultats sauvegardés : $csv_path")
println("═"^55)
