using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV

Random.seed!(42)
const MOI = JuMP.MOI

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n_values      = [2, 3, 10, 20]
bruit_niveaux = [0.0, 0.01, 0.05, 0.10]
dim           = 2
c6            = 1.0
d_min         = 1e-4
L             = 10.0

function nb_restarts_bench(n)
    n <= 3  ? 5 : n <= 10 ? 5 : n <= 20 ? 3 : 2
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

function ajouter_bruit(Q, σ, n)
    Q_bruit = copy(Q)
    for i in 1:n, j in i+1:n
        Q_bruit[i,j] = max(Q[i,j] + Q[i,j]*σ*randn(), Q[i,j]*0.01)
        Q_bruit[j,i] = Q_bruit[i,j]
    end
    return Q_bruit
end

# ═══════════════════════════════════════════════════════════════
#  §3  MDS EXACT
# ═══════════════════════════════════════════════════════════════
function solve_mds(n, dim, Q, c6)
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        d_ij = (c6/Q[i,j])^(1/6)
        D[i,j] = d_ij; D[j,i] = d_ij
    end
    B = Symmetric(-0.5 * (I - ones(n,n)/n) * D.^2 * (I - ones(n,n)/n))
    F = eigen(B)
    idx = sortperm(F.values, rev=true)[1:dim]
    X  = F.vectors[:, idx] * Diagonal(sqrt.(max.(F.values[idx], 0.0)))
    for d in 1:dim; X[:,d] .-= minimum(X[:,d]); end
    return X
end

# ═══════════════════════════════════════════════════════════════
#  §4  FORMULATION POLAIRE (référence NLP)
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)

    @variable(model, x[1:n, 1:dim])
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)
    @variable(model, R[i=1:n, j=i+1:n] >= d_min)

    for i in 1:n, d in 1:dim; set_start_value(x[i,d], x_init[i,d]); end
    for i in 1:n, j in i+1:n
        dx = x_init[i,1]-x_init[j,1]; dy = x_init[i,2]-x_init[j,2]
        set_start_value(theta[i,j], atan(dy,dx))
        set_start_value(R[i,j], max(norm([dx,dy]), d_min))
    end
    @variable(model, ep[i=1:n, j=i+1:n])
    for i in 1:n, j in i+1:n
        @constraint(model, x[i,1]-x[j,1] == R[i,j]*cos(theta[i,j]))
        @constraint(model, x[i,2]-x[j,2] == R[i,j]*sin(theta[i,j]))
        @constraint(model, ep[i,j] == (c6/R[i,j]^6)-Q[i,j])
    end
    @objective(model, Min, sum(ep[i,j]^2 for i in 1:n for j in i+1:n))

    optimize!(model)
    s = termination_status(model)
    s in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,
          MOI.SLOW_PROGRESS, MOI.ITERATION_LIMIT] &&
        return (s, objective_value(model), value.(x))
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §5  FORMULATION ANGULAIRE (nouvelle)
#
#  Idée : travailler directement sur (R_{ij}, θ_{ij}) sans x.
#
#  Variables : R_{ij} >= d_min,  θ_{ij} ∈ [-π, π]
#
#  Objectif :
#    min Σ_{i<j} (c6/R_{ij}^6 - Q_{ij})²
#
#  Contraintes de cohérence géométrique (hub = atome 1) :
#    Pour tout j < k (j,k > 1) :
#    z_{1k} = z_{1j} + z_{jk}
#    ⟺  R_{1k}cos θ_{1k} = R_{1j}cos θ_{1j} + R_{jk}cos θ_{jk}
#        R_{1k}sin θ_{1k} = R_{1j}sin θ_{1j} + R_{jk}sin θ_{jk}
#
#  Ces (n-1)(n-2) contraintes (linéaires en cos/sin, non-linéaires en R)
#  garantissent que les R_{ij}, θ_{ij} correspondent à une vraie
#  configuration 2D — sans jamais manipuler les x explicitement.
#
#  Avantage vs MDS : R_{ij} sont optimisés (robuste au bruit)
#  Avantage vs Polaire : pas de variables x, objectif direct sur R
#
#  Reconstruction des positions après résolution :
#    Fixer x_1 = (0,0) par invariance de translation
#    x_j = -R_{1j}(cos θ_{1j}, sin θ_{1j})  pour j=2,...,n
# ═══════════════════════════════════════════════════════════════
function solve_angulaire(n, dim, Q, c6, L, x_init)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)

    # Variables : pas de x — uniquement R et θ
    @variable(model, R[i=1:n, j=i+1:n] >= d_min)
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)

    # Warm-start depuis x_init
    for i in 1:n, j in i+1:n
        dx = x_init[i,1] - x_init[j,1]
        dy = x_init[i,2] - x_init[j,2]
        set_start_value(R[i,j],     max(norm([dx,dy]), d_min))
        set_start_value(theta[i,j], atan(dy, dx))
    end

    # Contraintes de cohérence géométrique
    # Pour tout j,k avec 1 < j < k :
    # z_{1k} = z_{1j} + z_{jk}  →  2 équations (cos et sin)
    for j in 2:n, k in j+1:n
        @constraint(model,
            R[1,k]*cos(theta[1,k]) ==
            R[1,j]*cos(theta[1,j]) + R[j,k]*cos(theta[j,k]))
        @constraint(model,
            R[1,k]*sin(theta[1,k]) ==
            R[1,j]*sin(theta[1,j]) + R[j,k]*sin(theta[j,k]))
    end

    # Objectif : erreur Rydberg directement sur R
    @objective(model, Min,
        sum((c6/R[i,j]^6 - Q[i,j])^2 for i in 1:n for j in i+1:n))

    optimize!(model)
    s = termination_status(model)

    if s in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,
             MOI.SLOW_PROGRESS, MOI.ITERATION_LIMIT]
        R_sol = value.(R)
        θ_sol = value.(theta)

        # Reconstruction : x_1=(0,0), x_j = -R_{1j}(cosθ_{1j}, sinθ_{1j})
        x_sol = zeros(n, 2)
        for j in 2:n
            x_sol[j,1] = -R_sol[1,j] * cos(θ_sol[1,j])
            x_sol[j,2] = -R_sol[1,j] * sin(θ_sol[1,j])
        end

        return (s, objective_value(model), x_sol)
    end
    return nothing
end

# ═══════════════════════════════════════════════════════════════
#  §6  RUNNER
# ═══════════════════════════════════════════════════════════════
function run_nlp(solver_fn, n, dim, Q, c6, L, x_inits, multistart)
    t0 = time()
    best_ry = Inf; best_st = "ÉCHEC"; best_x = nothing
    n_runs = multistart ? length(x_inits) : 1
    for k in 1:n_runs
        res = solver_fn(n, dim, Q, c6, L, x_inits[k])
        if res !== nothing
            stat, obj, x_sol = res
            ry = erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
            if ry < best_ry
                best_ry = ry; best_st = string(stat); best_x = x_sol
            end
            best_ry < 1e-12 && break
        end
    end
    return (best_ry, best_st, best_x, time()-t0)
end

# ═══════════════════════════════════════════════════════════════
#  §7  BENCHMARK — Q parfait ET Q bruité
# ═══════════════════════════════════════════════════════════════
function run_benchmark(n_values, bruit_niveaux, dim, c6, L)
    df = DataFrame(
        n=Int[], sigma=Float64[], formulation=String[],
        err_vs_vrai=Float64[], time_s=Float64[]
    )

    for n in n_values
        println("\n" * "═"^65)
        @printf("  n = %d\n", n)
        println("═"^65)

        x_vrais   = generer_init(n, dim, L)
        Q_parfait = zeros(n, n)
        for i in 1:n, j in i+1:n
            d = max(norm(x_vrais[i,:]-x_vrais[j,:]), 1e-5)
            Q_parfait[i,j] = Q_parfait[j,i] = c6/d^6
        end

        for σ in bruit_niveaux
            @printf("\n  ── σ = %.0f%% ──\n", σ*100)

            Q_bruit        = ajouter_bruit(Q_parfait, σ, n)
            n_rs           = nb_restarts_bench(n)
            x_inits        = [generer_init(n, dim, L) for _ in 1:n_rs]

            # MDS
            t_mds = @elapsed x_mds = solve_mds(n, dim, Q_bruit, c6)
            ev_mds = erreur_rydberg_vraie(x_mds, Q_parfait, c6, n, dim)
            @printf("    [%-12s] err_vrai=%+.3e  t=%.5fs\n",
                    "MDS", ev_mds, t_mds)
            push!(df, (n, σ, "MDS", ev_mds, t_mds))

            # Polaire
            ry_pol, st_pol, x_pol, t_pol =
                run_nlp(solve_polaire, n, dim, Q_bruit, c6, L, x_inits, true)
            ev_pol = erreur_rydberg_vraie(x_pol, Q_parfait, c6, n, dim)
            @printf("    [%-12s] err_vrai=%+.3e  t=%.2fs  [%s]\n",
                    "Polaire", ev_pol, t_pol, st_pol[1:min(20,length(st_pol))])
            push!(df, (n, σ, "Polaire", ev_pol, t_pol))

            # Angulaire
            ry_ang, st_ang, x_ang, t_ang =
                run_nlp(solve_angulaire, n, dim, Q_bruit, c6, L, x_inits, true)
            ev_ang = erreur_rydberg_vraie(x_ang, Q_parfait, c6, n, dim)
            @printf("    [%-12s] err_vrai=%+.3e  t=%.2fs  [%s]\n",
                    "Angulaire", ev_ang, t_ang, st_ang[1:min(20,length(st_ang))])
            push!(df, (n, σ, "Angulaire", ev_ang, t_ang))
        end
    end
    return df
end

const bench = run_benchmark(n_values, bruit_niveaux, dim, c6, L)

# ═══════════════════════════════════════════════════════════════
#  §8  TABLEAU FINAL
# ═══════════════════════════════════════════════════════════════
println("\n\n" * "═"^70)
println("  COMPARAISON MDS / Polaire / Angulaire")
println("═"^70)

for grp in groupby(bench, [:n, :sigma])
    r = grp[1,:]
    @printf("\n  n=%-3d  σ=%.0f%%\n", r.n, r.sigma*100)
    min_ev = minimum(replace(grp.err_vs_vrai, Inf => 1e300))
    for row in eachrow(grp)
        ev_cmp = isinf(row.err_vs_vrai) ? 1e300 : row.err_vs_vrai
        marker = ev_cmp == min_ev ? " ← ★" : ""
        @printf("    %-12s  err_vrai=%+.4e  t=%7.4fs%s\n",
                row.formulation, row.err_vs_vrai, row.time_s, marker)
    end
end

csv_path = joinpath(@__DIR__, "benchmark_angulaire.csv")
CSV.write(csv_path, bench)
println("\n\nRésultats sauvegardés : $csv_path")
println("═"^70)