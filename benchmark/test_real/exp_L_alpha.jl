# ═══════════════════════════════════════════════════════════════
#  Expériences L variable + α L
#
#  zmin_opt = (c6 / Q_max)^{1/6}   (distance minimale optimale)
#
#  Expérience 1 — avec zmin :
#      min f(x) + α L
#      s.c.  0 ≤ x ≤ L,  ‖x_i - x_j‖ ≥ zmin_opt
#      → observer le L trouvé par Ipopt
#
#  Expérience 2 — sans zmin :
#      min f(x) + α L
#      s.c.  0 ≤ x ≤ L   (pas de contrainte de distance)
#      → vérifier si min_{i<j} ‖x_i-x_j‖ ≈ zmin_opt
#
#  Lancer :
#      julia --project=. test_real/exp_L_alpha.jl
# ═══════════════════════════════════════════════════════════════

using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV, Statistics

include(joinpath(@__DIR__, "load_config_L_alpha.jl"))
const CFG = load_L_alpha_config(; profile=:simple)

const MOI = JuMP.MOI
const c6 = CFG.c6
const dim = CFG.dim

# Compatibilité avec generer_init (seuil par défaut = z_min)
const z_min = CFG.z_min
const d_min = CFG.d_min

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "formulations.jl"))   # MDS + solveurs classiques
include(joinpath(@__DIR__, "formulations_Lvar.jl"))

const PASQAL_ROOT = CFG.pasqal_root
const OUT_CSV = joinpath(CFG.out_dir, CFG.out_csv_name)

# ── Paramètres expérimentaux (voir config_L_alpha.toml) ───────
const N_VALUES = CFG.n_values
const N_INSTANCES = CFG.n_instances
const ALPHAS = CFG.alphas
const L_INIT = CFG.l_init

function initial_points(n, dim, Q, c6, L_guess)
    pts = Matrix{Float64}[]
    # MDS si possible, sinon aléatoire
    try
        res = solve_mds(n, dim, Q, c6)
        if res !== nothing
            _, x_mds = res
            x = copy(x_mds)
            # ramener dans [0, L_guess]
            for d in 1:dim
                mn = minimum(x[:, d])
                x[:, d] .-= mn
            end
            mx = maximum(x)
            if mx > L_guess && mx > 0
                x .*= (L_guess / mx)
            end
            push!(pts, x)
        end
    catch
    end
    push!(pts, generer_init(n, dim, L_guess; seuil=z_min))
    return pts
end

function run_one(solver_fn, n, dim, Q, c6, x_inits;
                 alpha::Float64, zmin, L_init::Float64)
    best = nothing
    best_ry = Inf
    t0 = time()
    for x0 in x_inits
        res = solver_fn(n, dim, Q, c6, x0;
                        alpha=alpha, zmin=zmin, L_init=L_init,
                        L_lb=zmin === nothing ? 1e-3 : Float64(zmin))
        res === nothing && continue
        status, obj, x_sol, L_sol = res
        ry = erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
        if ry < best_ry
            best_ry = ry
            dmin = min_pairwise_distance(x_sol)
            best = (
                status = string(status),
                obj = obj,
                err_rydberg = ry,
                L_found = L_sol,
                dmin_found = dmin,
                x = x_sol,
            )
        end
        best_ry < 1e-12 && break
    end
    return best, time() - t0
end

function make_row(; exp_name, instance, n, alpha, formulation,
                    zmin_opt, use_zmin, result, time_s)
    if result === nothing
        return (
            exp = exp_name, instance=instance, n=n, alpha=alpha,
            formulation=formulation, zmin_opt=zmin_opt, use_zmin=use_zmin,
            L_found=NaN, dmin_found=NaN, ratio_dmin_zmin=NaN,
            err_rydberg=Inf, obj=NaN, status="ECHEC", time_s=time_s,
        )
    end
    ratio = result.dmin_found / zmin_opt
    return (
        exp = exp_name, instance=instance, n=n, alpha=alpha,
        formulation=formulation, zmin_opt=zmin_opt, use_zmin=use_zmin,
        L_found=result.L_found, dmin_found=result.dmin_found,
        ratio_dmin_zmin=ratio,
        err_rydberg=result.err_rydberg, obj=result.obj,
        status=result.status, time_s=time_s,
    )
end

function run_experiments(;
    n_values=N_VALUES,
    n_instances=N_INSTANCES,
    alphas=ALPHAS,
    formulations=filter(p -> first(p) in Set(split(CFG.forms, ",")), collect(LVAR_FORMULATIONS)),
    seed=CFG.seed,
)
    Random.seed!(seed)
    rows = NamedTuple[]

    println("═"^72)
    println("  EXP L-variable : zmin_opt = (c6/Qmax)^(1/6),  obj += α·L")
    println("  CONFIG=$(CFG.config_path)")
    println("═"^72)

    for inst in 1:n_instances
        println("\n── Instance $inst / $n_instances ──")
        for n in n_values
            # Q exacte synthétique (référence géométrique)
            x_ref = generer_init(n, dim, L_INIT; seuil=z_min)
            Q = build_Q_matrix(x_ref, c6, n)
            zmin_opt = zmin_optimal(Q, c6)
            x_inits = initial_points(n, dim, Q, c6, L_INIT)

            @printf("  n=%-3d  zmin_opt=%.6f  (c6/Qmax)^(1/6)\n", n, zmin_opt)

            for alpha in alphas
                for (fname, solver_fn) in formulations
                    # ── Expérience 1 : avec zmin ─────────────────
                    res1, t1 = run_one(solver_fn, n, dim, Q, c6, x_inits;
                                       alpha=alpha, zmin=zmin_opt, L_init=L_INIT)
                    push!(rows, make_row(
                        exp_name="avec_zmin", instance=inst, n=n, alpha=alpha,
                        formulation=fname, zmin_opt=zmin_opt, use_zmin=true,
                        result=res1, time_s=t1,
                    ))
                    if res1 !== nothing
                        ok1 = abs(res1.dmin_found - zmin_opt) / zmin_opt < 0.2 ?
                              "≈zmin ✓" : "≠zmin"
                        @printf("    [avec zmin] α=%-8.1e %-14s  L*=%.4f  dmin=%.4f  zmin_opt=%.4f  ratio=%.3f %s  obj=%.2e  ry=%.2e  [%s]\n",
                                alpha, fname, res1.L_found, res1.dmin_found, zmin_opt,
                                res1.dmin_found / zmin_opt, ok1, res1.obj, res1.err_rydberg, res1.status)
                    else
                        @printf("    [avec zmin] α=%-8.1e %-14s  ÉCHEC\n", alpha, fname)
                    end

                    # ── Expérience 2 : sans zmin ─────────────────
                    res2, t2 = run_one(solver_fn, n, dim, Q, c6, x_inits;
                                       alpha=alpha, zmin=nothing, L_init=L_INIT)
                    push!(rows, make_row(
                        exp_name="sans_zmin", instance=inst, n=n, alpha=alpha,
                        formulation=fname, zmin_opt=zmin_opt, use_zmin=false,
                        result=res2, time_s=t2,
                    ))
                    if res2 !== nothing
                        ok = abs(res2.dmin_found - zmin_opt) / zmin_opt < 0.2 ?
                             "≈zmin ✓" : "≠zmin"
                        @printf("    [sans zmin] α=%-8.1e %-14s  L*=%.4f  dmin=%.4f  zmin_opt=%.4f  ratio=%.3f %s  obj=%.2e  ry=%.2e  [%s]\n",
                                alpha, fname, res2.L_found, res2.dmin_found,
                                zmin_opt, res2.dmin_found / zmin_opt, ok, res2.obj, res2.err_rydberg, res2.status)
                    else
                        @printf("    [sans zmin] α=%-8.1e %-14s  ÉCHEC\n", alpha, fname)
                    end
                end
            end
        end
    end

    df = DataFrame(rows)
    CSV.write(OUT_CSV, df)
    println("\nRésultats sauvegardés : $OUT_CSV")

    # ── Synthèse ──────────────────────────────────────────────
    println("\n" * "═"^72)
    println("  SYNTHÈSE")
    println("═"^72)

    for exp_name in ("avec_zmin", "sans_zmin")
        sub = filter(r -> r.exp == exp_name, df)
        println("\n▶ Expérience : $exp_name")
        for g in groupby(sub, [:n, :alpha, :formulation])
            r0 = g[1, :]
            L_m = mean(filter(isfinite, g.L_found))
            d_m = mean(filter(isfinite, g.dmin_found))
            ratio_m = mean(filter(isfinite, g.ratio_dmin_zmin))
            ry_m = mean(filter(isfinite, g.err_rydberg))
            @printf("  n=%-3d α=%-8.1e %-14s  L̄=%.4f  dmin̄=%.4f  dmin/zmin̄=%.3f  rȳ=%.2e\n",
                    r0.n, r0.alpha, r0.formulation, L_m, d_m, ratio_m, ry_m)
        end
    end

    return df
end

# Lancer automatiquement si exécuté comme script
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiments()
end
