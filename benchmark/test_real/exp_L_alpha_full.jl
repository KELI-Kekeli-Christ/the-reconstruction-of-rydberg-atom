# ═══════════════════════════════════════════════════════════════
#  exp_L_alpha_full.jl
#
#  Ordre d'exécution :
#  Exp 3 — Pasqal (tous les .gml)
#  Exp 2 — Q exacte synthétique (x généré → Q)
#  Exp 1 — Q aléatoire (Q_bruit_dist)
#
#  Scaling après chaque solve :
#    λ = L_CHOSEN / L*
#    si L_CHOSEN < L* : x′=λx*, zmin′=λ zmin, Q′=Q/λ^6
#    si L_CHOSEN ≥ L* : pas de scaling
#
#  julia --project=. test_real/exp_L_alpha_full.jl
# ═══════════════════════════════════════════════════════════════

using Dates
using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV, Statistics

include(joinpath(@__DIR__, "load_config_L_alpha.jl"))
const CFG = load_L_alpha_config(; profile=:full)

const MOI = JuMP.MOI
const c6 = CFG.c6
const dim = CFG.dim
const z_min = CFG.z_min
const d_min = CFG.d_min

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "formulations.jl"))
include(joinpath(@__DIR__, "formulations_Lvar.jl"))

const PASQAL_ROOT = CFG.pasqal_root
const OUT_DIR = CFG.out_dir
const OUT_CSV = CFG.out_csv
const OUT_LOG = CFG.out_log

const N_VALUES = CFG.n_values
const N_INSTANCES = CFG.n_instances
const ALPHAS = CFG.alphas
const L_INIT = CFG.l_init
const L_CHOSEN = CFG.l_chosen
const RUN_EXP1 = CFG.run_exp1
const RUN_EXP2 = CFG.run_exp2
const RUN_EXP3 = CFG.run_exp3
const FORM_FILTER = CFG.forms

# Tolérance relative : |dmin - zmin_opt| / zmin_opt < TOL ⇒ "≈zmin ✓"
const ZMIN_MATCH_TOL = CFG.zmin_match_tol

const COLS = [
    :exp, :source, :instance, :folder, :file, :n, :n_edges,
    :alpha, :formulation, :use_zmin, :zmin_opt,
    :L_init, :L_chosen, :L_found, :dmin_found, :ratio_dmin_zmin,
    :rel_err_dmin_zmin, :dmin_matches_zmin, :zmin_check,
    :err_rydberg, :obj, :status, :time_s,
    :scale_applied, :lambda, :zmin_scaled, :dmin_scaled,
    :rel_err_dmin_zmin_scaled, :dmin_matches_zmin_scaled, :zmin_check_scaled,
    :err_rydberg_scaled, :scale_note, :c6, :dim, :seed_note,
]

function zmin_check_label(dmin, zmin_opt; tol=ZMIN_MATCH_TOL)
    if !(isfinite(dmin) && isfinite(zmin_opt) && zmin_opt > 0)
        return (rel_err=NaN, matches=false, label="n/a")
    end
    rel = abs(dmin - zmin_opt) / zmin_opt
    ok = rel < tol
    return (rel_err=rel, matches=ok, label=ok ? "≈zmin ✓" : "≠zmin")
end

function selected_formulations()
    wanted = Set(split(FORM_FILTER, ","))
    return filter(p -> first(p) in wanted, collect(LVAR_FORMULATIONS))
end

function pasqal_gml_paths()
    instances = NamedTuple[]
    for folder in ("freespace", "sqauare", "triangle")
        folder_path = joinpath(PASQAL_ROOT, folder)
        isdir(folder_path) || continue
        for f in sort(readdir(folder_path))
            endswith(lowercase(f), ".gml") || continue
            push!(instances, (folder=folder, name=f, path=joinpath(folder_path, f)))
        end
    end
    return instances
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
                status = string(status),
                obj = obj,
                err_rydberg = ry,
                L_found = L_sol,
                dmin_found = min_pairwise_distance(x_sol),
                x = x_sol,
            )
        end
        best_ry < 1e-12 && break
    end
    return best, time() - t0
end

function apply_scaling(x_star, Q, zmin_opt, L_star, L_chosen)
    if L_chosen >= L_star - 1e-12
        return (
            scaled = false,
            lambda = 1.0,
            zmin = zmin_opt,
            ry = erreur_rydberg_vraie(x_star, Q, c6, size(x_star, 1), dim),
            dmin = min_pairwise_distance(x_star),
            note = "L_choisi>=L* : pas de scaling",
        )
    end
    λ = L_chosen / L_star
    x_p = x_star .* λ
    Q_p = Q ./ (λ^6)
    zmin_p = zmin_opt * λ
    return (
        scaled = true,
        lambda = λ,
        zmin = zmin_p,
        ry = erreur_rydberg_vraie(x_p, Q_p, c6, size(x_p, 1), dim),
        dmin = min_pairwise_distance(x_p),
        note = "L_choisi<L* : x'=λx, zmin'=λzmin, Q'=Q/λ^6",
    )
end

function make_row(; source, instance, folder, file, n, n_edges, alpha, formulation,
                   use_zmin, zmin_opt, result, time_s, Q, seed_note)
    if result === nothing
        return (;
            exp = string(source, "/", use_zmin ? "avec_zmin" : "sans_zmin"),
            source, instance, folder, file, n, n_edges,
            alpha, formulation, use_zmin, zmin_opt,
            L_init = L_INIT, L_chosen = L_CHOSEN,
            L_found = NaN, dmin_found = NaN, ratio_dmin_zmin = NaN,
            rel_err_dmin_zmin = NaN, dmin_matches_zmin = false, zmin_check = "ECHEC",
            err_rydberg = Inf, obj = NaN, status = "ECHEC", time_s,
            scale_applied = false, lambda = NaN,
            zmin_scaled = NaN, dmin_scaled = NaN,
            rel_err_dmin_zmin_scaled = NaN, dmin_matches_zmin_scaled = false,
            zmin_check_scaled = "ECHEC",
            err_rydberg_scaled = NaN, scale_note = "echec solve",
            c6, dim, seed_note,
        )
    end
    sc = apply_scaling(result.x, Q, zmin_opt, result.L_found, L_CHOSEN)
    # Vérif principale : dmin(x*) à L* ≟ zmin_opt = (c6/Qmax)^{1/6}
    chk = zmin_check_label(result.dmin_found, zmin_opt)
    # Après scaling : dmin(x') ≟ zmin' = λ zmin_opt
    chk_s = zmin_check_label(sc.dmin, sc.zmin)
    return (;
        exp = string(source, "/", use_zmin ? "avec_zmin" : "sans_zmin"),
        source, instance, folder, file, n, n_edges,
        alpha, formulation, use_zmin, zmin_opt,
        L_init = L_INIT, L_chosen = L_CHOSEN,
        L_found = result.L_found,
        dmin_found = result.dmin_found,
        ratio_dmin_zmin = result.dmin_found / zmin_opt,
        rel_err_dmin_zmin = chk.rel_err,
        dmin_matches_zmin = chk.matches,
        zmin_check = chk.label,
        err_rydberg = result.err_rydberg,
        obj = result.obj,
        status = result.status,
        time_s,
        scale_applied = sc.scaled,
        lambda = sc.lambda,
        zmin_scaled = sc.zmin,
        dmin_scaled = sc.dmin,
        rel_err_dmin_zmin_scaled = chk_s.rel_err,
        dmin_matches_zmin_scaled = chk_s.matches,
        zmin_check_scaled = chk_s.label,
        err_rydberg_scaled = sc.ry,
        scale_note = sc.note,
        c6, dim, seed_note,
    )
end

function flush_csv!(rows)
    isempty(rows) && return
    CSV.write(OUT_CSV, DataFrame(rows))
end

function logmsg(io, s)
    println(io, s); flush(io)
    println(s); flush(stdout)
end

function run_block!(rows, logio; source, instance, folder, file, n, Q,
                    alpha, formulations, x_inits, seed_note="")
    zmin_opt = zmin_optimal(Q, c6)
    n_edges = length(active_pairs(Q))
    for (fname, solver_fn) in formulations
        for use_zmin in (true, false)
            tag = use_zmin ? "avec_zmin" : "sans_zmin"
            zarg = use_zmin ? zmin_opt : nothing
            res, t = try
                run_one(solver_fn, n, dim, Q, c6, x_inits;
                        alpha=alpha, zmin=zarg, L_init=L_INIT)
            catch err
                logmsg(logio, "    ERREUR $fname: $err")
                nothing, 0.0
            end
            row = make_row(;
                source, instance, folder, file, n, n_edges, alpha,
                formulation=fname, use_zmin, zmin_opt,
                result=res, time_s=t, Q=Q, seed_note=seed_note,
            )
            push!(rows, row)
            if res === nothing
                logmsg(logio, @sprintf("    [%s] α=%-8.1e %-14s ÉCHEC", tag, alpha, fname))
            else
                # Affiche : L*, dmin(L*), zmin_opt, vérif dmin≟zmin_opt (+ scaling)
                logmsg(logio, @sprintf(
                    "    [%s] α=%-8.1e %-14s L*=%.4f dmin=%.4f zmin_opt=%.4f ratio=%.3f %s  obj=%.2e  ry=%.2e | scale: dmin'=%.4f zmin'=%.4f %s ry'=%.2e [%s]",
                    tag, alpha, fname, res.L_found, res.dmin_found, zmin_opt,
                    row.ratio_dmin_zmin, row.zmin_check, res.obj, res.err_rydberg,
                    row.dmin_scaled, row.zmin_scaled, row.zmin_check_scaled,
                    row.err_rydberg_scaled, res.status))
            end
        end
        flush_csv!(rows)
    end
end

function run_all(; seed=CFG.seed)
    mkpath(OUT_DIR)
    open(OUT_LOG, "w") do logio
        Random.seed!(seed)
        rows = Any[]
        formulations = selected_formulations()

        logmsg(logio, "═"^72)
        logmsg(logio, "  EXP L-variable FULL — $(Dates.format(Dates.now(), dateformat"yyyy-mm-dd HH:MM:SS"))")
        logmsg(logio, "═"^72)
        logmsg(logio, "CONFIG=$(CFG.config_path)")
        logmsg(logio, "OUT_CSV=$OUT_CSV")
        logmsg(logio, "PASQAL_ROOT=$PASQAL_ROOT")
        logmsg(logio, "N_VALUES=$N_VALUES N_INSTANCES=$N_INSTANCES")
        logmsg(logio, "ALPHAS=$ALPHAS L_INIT=$L_INIT L_CHOSEN=$L_CHOSEN")
        logmsg(logio, "ZMIN_MATCH_TOL=$ZMIN_MATCH_TOL  (dmin≈zmin_opt si |dmin-zmin|/zmin < tol)")
        logmsg(logio, "FORMS=$(join(first.(formulations), ", "))")
        logmsg(logio, "RUN_EXP1=$RUN_EXP1 RUN_EXP2=$RUN_EXP2 RUN_EXP3=$RUN_EXP3")

        # Ordre demandé : Pasqal → synthétique → aléatoire
        if RUN_EXP3
            instances = pasqal_gml_paths()
            logmsg(logio, "\n▶ EXP 3 — Pasqal ($(length(instances)) .gml)")
            for (k, inst) in enumerate(instances)
                Q = read_gml_q(inst.path)
                n = size(Q, 1)
                if isempty(active_pairs(Q))
                    logmsg(logio, "  skip $(inst.name)"); continue
                end
                x_inits = initial_points(n, dim, Q, c6, L_INIT)
                logmsg(logio, @sprintf("\n── Exp3 [%d/%d] %s/%s n=%d zmin=%.4f ──",
                       k, length(instances), inst.folder, inst.name, n, zmin_optimal(Q, c6)))
                for alpha in ALPHAS
                    run_block!(rows, logio; source="exp3_pasqal", instance=k,
                               folder=inst.folder, file=inst.name, n=n, Q=Q, alpha=alpha,
                               formulations=formulations, x_inits=x_inits, seed_note="pasqal")
                end
            end
        end

        if RUN_EXP2
            logmsg(logio, "\n▶ EXP 2 — Q exacte synthétique")
            for inst in 1:N_INSTANCES, n in N_VALUES
                x_ref = generer_init(n, dim, L_INIT; seuil=z_min)
                Q = build_Q_matrix(x_ref, c6, n)
                x_inits = initial_points(n, dim, Q, c6, L_INIT)
                logmsg(logio, @sprintf("\n── Exp2 inst=%d n=%d zmin=%.4f ──", inst, n, zmin_optimal(Q, c6)))
                for alpha in ALPHAS
                    run_block!(rows, logio; source="exp2_Q_exacte_synthetique", instance=inst,
                               folder="", file="x_genere", n=n, Q=Q, alpha=alpha,
                               formulations=formulations, x_inits=x_inits, seed_note="seed=$seed")
                end
            end
        end

        if RUN_EXP1
            logmsg(logio, "\n▶ EXP 1 — Q aléatoire")
            for inst in 1:N_INSTANCES, n in N_VALUES
                Q = Q_bruit_dist(n, c6, L_INIT)
                x_inits = initial_points(n, dim, Q, c6, L_INIT)
                logmsg(logio, @sprintf("\n── Exp1 inst=%d n=%d zmin=%.4f ──", inst, n, zmin_optimal(Q, c6)))
                for alpha in ALPHAS
                    run_block!(rows, logio; source="exp1_Q_aleatoire", instance=inst,
                               folder="", file="Q_bruit_dist", n=n, Q=Q, alpha=alpha,
                               formulations=formulations, x_inits=x_inits, seed_note="seed=$seed")
                end
            end
        end

        flush_csv!(rows)
        logmsg(logio, "\nTERMINÉ — $(length(rows)) lignes → $OUT_CSV")
        return DataFrame(rows)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
