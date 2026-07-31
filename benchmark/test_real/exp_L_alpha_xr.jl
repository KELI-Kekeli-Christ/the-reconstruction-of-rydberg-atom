# ═══════════════════════════════════════════════════════════════
#  exp_L_alpha_xr.jl
#
#  Export x* et r_ij dans des CSV (n=4,10).
#  4 expériences :
#    1) Exp2 synthétique + avec zmin
#    2) Exp2 synthétique + sans zmin
#    3) Exp1 Q aléatoire + avec zmin
#    4) Exp1 Q aléatoire + sans zmin
#
#  Sorties :
#    exp_L_alpha_xr_runs.csv   — 1 ligne / solve (x aplatis)
#    exp_L_alpha_xr_pairs.csv  — 1 ligne / paire (i,j,r,d,Q,...)
#
#  julia test_real/exp_L_alpha_xr.jl
# ═══════════════════════════════════════════════════════════════

using Dates
using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV, Statistics

include(joinpath(@__DIR__, "load_config_L_alpha.jl"))
const CFG = load_L_alpha_config(; profile=:xr)

const MOI = JuMP.MOI
const c6 = CFG.c6
const dim = CFG.dim
const z_min = CFG.z_min
const d_min = CFG.d_min

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "formulations.jl"))
include(joinpath(@__DIR__, "formulations_Lvar.jl"))

const OUT_DIR = CFG.out_dir
const OUT_RUNS = CFG.out_runs
const OUT_PAIRS = CFG.out_pairs
const OUT_LOG = CFG.out_log

const N_VALUES = CFG.n_values
const N_INSTANCES = CFG.n_instances
const ALPHAS = CFG.alphas
const L_INIT = CFG.l_init
const FORM_FILTER = CFG.forms
const ZMIN_MATCH_TOL = CFG.zmin_match_tol

function selected_formulations()
    wanted = Set(split(FORM_FILTER, ","))
    return filter(p -> first(p) in wanted, collect(LVAR_FORMULATIONS))
end

function format_x(x::AbstractMatrix)
    # "x1=(a,b); x2=(c,d); ..."
    parts = String[]
    for i in 1:size(x, 1)
        push!(parts, @sprintf("x%d=(%.8g,%.8g)", i, x[i, 1], x[i, 2]))
    end
    return join(parts, "; ")
end

function format_x_flat(x::AbstractMatrix)
    # CSV-friendly: x11,x12,x21,x22,...
    vals = String[]
    for i in 1:size(x, 1), d in 1:size(x, 2)
        push!(vals, @sprintf("%.10g", x[i, d]))
    end
    return join(vals, ",")
end

function pairwise_rows(x, Q)
    n = size(x, 1)
    rows = NamedTuple[]
    for i in 1:n, j in i+1:n
        rij = norm(x[i, :] - x[j, :])
        dij = (c6 / max(Q[i, j], 1e-300))^(1 / 6)
        Qhat = c6 / max(rij, 1e-30)^6
        push!(rows, (
            i=i, j=j, rij=rij, dij=dij, abs_r_minus_d=abs(rij - dij),
            Qij=Q[i, j], Qhat=Qhat, deltaQ2=(Q[i, j] - Qhat)^2,
            taylor_weight=36 / max(dij, 1e-30)^14,
            collapsed=rij < 1e-4,
        ))
    end
    return rows
end

function initial_points(n, dim, Q, c6, L_guess)
    pts = Matrix{Float64}[]
    try
        res = solve_mds(n, dim, Q, c6)
        if res !== nothing
            _, x_mds = res
            x = copy(x_mds)
            for d in 1:dim
                x[:, d] .-= minimum(x[:, d])
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

function run_one(solver_fn, n, dim, Q, c6, x_inits; alpha::Float64, zmin, L_init::Float64)
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
            best = (
                status=string(status), obj=obj, err_rydberg=ry,
                L_found=L_sol, dmin_found=min_pairwise_distance(x_sol), x=x_sol,
            )
        end
        best_ry < 1e-12 && break
    end
    return best, time() - t0
end

function logmsg(io, s)
    println(io, s); flush(io)
    println(s); flush(stdout)
end

function flush_all!(run_rows, pair_rows)
    CSV.write(OUT_RUNS, DataFrame(run_rows))
    CSV.write(OUT_PAIRS, DataFrame(pair_rows))
end

function run_block!(run_rows, pair_rows, logio; exp_id, source, instance, n, Q,
                    alpha, formulations, x_inits, seed_note="")
    zmin_opt = zmin_optimal(Q, c6)
    n_edges = length(active_pairs(Q))
    for (fname, solver_fn) in formulations
        for use_zmin in (true, false)
            tag = use_zmin ? "avec_zmin" : "sans_zmin"
            exp_name = "$(source)/$(tag)"
            zarg = use_zmin ? zmin_opt : nothing
            res, t = try
                run_one(solver_fn, n, dim, Q, c6, x_inits;
                        alpha=alpha, zmin=zarg, L_init=L_INIT)
            catch err
                logmsg(logio, "    ERREUR $fname: $err")
                nothing, 0.0
            end

            if res === nothing
                push!(run_rows, (
                    exp_id=exp_id, exp=exp_name, source=source, instance=instance,
                    n=n, n_edges=n_edges, alpha=alpha, formulation=fname,
                    use_zmin=use_zmin, zmin_opt=zmin_opt,
                    L_found=NaN, dmin_found=NaN, ratio_dmin_zmin=NaN,
                    dmin_matches_zmin=false, zmin_check="ECHEC",
                    err_rydberg=Inf, obj=NaN, status="ECHEC", time_s=t,
                    n_collapsed_pairs=0, x_pretty="", x_flat="",
                    c6=c6, dim=dim, seed_note=seed_note,
                ))
                logmsg(logio, @sprintf("    [%s] α=%-8.1e %-14s ÉCHEC", tag, alpha, fname))
            else
                ratio = res.dmin_found / zmin_opt
                match = abs(res.dmin_found - zmin_opt) / zmin_opt < ZMIN_MATCH_TOL
                prs = pairwise_rows(res.x, Q)
                n_coll = count(p -> p.collapsed, prs)
                push!(run_rows, (
                    exp_id=exp_id, exp=exp_name, source=source, instance=instance,
                    n=n, n_edges=n_edges, alpha=alpha, formulation=fname,
                    use_zmin=use_zmin, zmin_opt=zmin_opt,
                    L_found=res.L_found, dmin_found=res.dmin_found,
                    ratio_dmin_zmin=ratio,
                    dmin_matches_zmin=match,
                    zmin_check=match ? "≈zmin ✓" : "≠zmin",
                    err_rydberg=res.err_rydberg, obj=res.obj,
                    status=res.status, time_s=t,
                    n_collapsed_pairs=n_coll,
                    x_pretty=format_x(res.x),
                    x_flat=format_x_flat(res.x),
                    c6=c6, dim=dim, seed_note=seed_note,
                ))
                for p in prs
                    push!(pair_rows, (
                        exp_id=exp_id, exp=exp_name, source=source, instance=instance,
                        n=n, alpha=alpha, formulation=fname, use_zmin=use_zmin,
                        zmin_opt=zmin_opt, L_found=res.L_found,
                        err_rydberg=res.err_rydberg, status=res.status,
                        i=p.i, j=p.j, rij=p.rij, dij=p.dij,
                        abs_r_minus_d=p.abs_r_minus_d,
                        Qij=p.Qij, Qhat_C6_over_r6=p.Qhat, deltaQ2=p.deltaQ2,
                        taylor_weight_36_d14=p.taylor_weight,
                        collapsed=p.collapsed,
                        xi1=res.x[p.i, 1], xi2=res.x[p.i, 2],
                        xj1=res.x[p.j, 1], xj2=res.x[p.j, 2],
                    ))
                end
                logmsg(logio, @sprintf(
                    "    [%s] α=%-8.1e %-14s L*=%.4f dmin=%.4e zmin=%.4e %s  obj=%.2e  Eryd=%.2e collapsed=%d [%s]",
                    tag, alpha, fname, res.L_found, res.dmin_found, zmin_opt,
                    match ? "≈zmin ✓" : "≠zmin", res.obj, res.err_rydberg, n_coll, res.status))
            end
            flush_all!(run_rows, pair_rows)
        end
    end
end

function run_all(; seed=CFG.seed)
    mkpath(OUT_DIR)
    open(OUT_LOG, "w") do logio
        Random.seed!(seed)
        run_rows = Any[]
        pair_rows = Any[]
        formulations = selected_formulations()

        logmsg(logio, "═"^72)
        logmsg(logio, "  EXPORT x + r_ij — $(Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        logmsg(logio, "═"^72)
        logmsg(logio, "CONFIG=$(CFG.config_path)")
        logmsg(logio, "OUT_RUNS=$OUT_RUNS")
        logmsg(logio, "OUT_PAIRS=$OUT_PAIRS")
        logmsg(logio, "N_VALUES=$N_VALUES N_INSTANCES=$N_INSTANCES ALPHAS=$ALPHAS")
        logmsg(logio, "FORMS=$(join(first.(formulations), ", "))")
        logmsg(logio, "4 expériences : Exp2±zmin, Exp1±zmin")

        # ── Exp 2 : Q exacte synthétique ──
        logmsg(logio, "\n▶ EXP 2 — Q exacte synthétique (exp_id=2)")
        for inst in 1:N_INSTANCES, n in N_VALUES
            x_ref = generer_init(n, dim, L_INIT; seuil=z_min)
            Q = build_Q_matrix(x_ref, c6, n)
            x_inits = initial_points(n, dim, Q, c6, L_INIT)
            logmsg(logio, @sprintf("\n── Exp2 inst=%d n=%d zmin=%.4f ──", inst, n, zmin_optimal(Q, c6)))
            for alpha in ALPHAS
                run_block!(run_rows, pair_rows, logio;
                           exp_id=2, source="exp2_Q_exacte_synthetique",
                           instance=inst, n=n, Q=Q, alpha=alpha,
                           formulations=formulations, x_inits=x_inits,
                           seed_note="seed=$seed")
            end
        end

        # ── Exp 1 : Q aléatoire ──
        logmsg(logio, "\n▶ EXP 1 — Q aléatoire (exp_id=1)")
        for inst in 1:N_INSTANCES, n in N_VALUES
            Q = Q_bruit_dist(n, c6, L_INIT)
            x_inits = initial_points(n, dim, Q, c6, L_INIT)
            logmsg(logio, @sprintf("\n── Exp1 inst=%d n=%d zmin=%.4f ──", inst, n, zmin_optimal(Q, c6)))
            for alpha in ALPHAS
                run_block!(run_rows, pair_rows, logio;
                           exp_id=1, source="exp1_Q_aleatoire",
                           instance=inst, n=n, Q=Q, alpha=alpha,
                           formulations=formulations, x_inits=x_inits,
                           seed_note="seed=$seed")
            end
        end

        flush_all!(run_rows, pair_rows)
        logmsg(logio, "\nTERMINÉ — $(length(run_rows)) runs, $(length(pair_rows)) paires")
        logmsg(logio, "  → $OUT_RUNS")
        logmsg(logio, "  → $OUT_PAIRS")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
