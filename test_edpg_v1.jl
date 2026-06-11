using JuMP, Ipopt, LinearAlgebra, Random, Printf, Plots

Random.seed!(42)

# ═════════════════════════════════════
# PARAMÈTRES
# ═════════════════════════════════════
n_values = [5, 8, 10, 15]
dim = 2
c6 = 1.0

noise_levels = [0.01, 0.05, 0.1, 0.5]

# ═════════════════════════════════════
# INIT
# ═════════════════════════════════════
function generer_init(n, dim, L)
    return L * rand(n, dim)
end

# ═════════════════════════════════════
# ERREUR RYDBERG
# ═════════════════════════════════════
function erreur_rydberg(x_sol, Q, c6)
    n = size(x_sol,1)
    err = 0.0
    for i in 1:n, j in i+1:n
        d = max(norm(x_sol[i,:] - x_sol[j,:]), 1e-6)
        err += (c6 / d^6 - Q[i,j])^2
    end
    return err
end

# ═════════════════════════════════════
# BRUIT SUR Q
# ═════════════════════════════════════
function bruiter_Q(Q, σ)
    n = size(Q,1)
    Qb = copy(Q)
    for i in 1:n, j in i+1:n
        noise = σ * randn()
        Qb[i,j] = max(Q[i,j] * (1 + noise), 1e-8)
        Qb[j,i] = Qb[i,j]
    end
    return Qb
end

# ═════════════════════════════════════
# STANDARD
# ═════════════════════════════════════
function solve_standard(n, dim, Q, c6, x_init)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variable(m, x[1:n,1:dim])

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end

    @expression(m, d2[i=1:n, j=i+1:n],
        sum((x[i,k]-x[j,k])^2 for k in 1:dim) + 1e-6
    )

    @objective(m, Min,
        sum((c6 / d2[i,j]^3 - Q[i,j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)

    return value.(x), objective_value(m), termination_status(m)
end

# ═════════════════════════════════════
# POLAIRE
# ═════════════════════════════════════
function solve_polaire(n, dim, Q, c6, x_init)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variable(m, x[1:n,1:dim])
    @variable(m, R[i=1:n,j=i+1:n] >= 1e-6)
    @variable(m, err[i=1:n,j=i+1:n])

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end

    for i in 1:n, j in i+1:n
        @constraint(m,
            (x[i,1]-x[j,1])^2 + (x[i,2]-x[j,2])^2 == R[i,j]^2
        )

        @constraint(m,
            err[i,j] == (c6 / (R[i,j]^6 + 1e-12)) - Q[i,j]
        )
    end

    @objective(m, Min, sum(err[i,j]^2 for i in 1:n for j in i+1:n))

    optimize!(m)

    return value.(x), objective_value(m), termination_status(m)
end

# ═════════════════════════════════════
# MDS
# ═════════════════════════════════════
function solve_mds(n, dim, Q, c6)
    D = zeros(n,n)
    for i in 1:n, j in i+1:n
        D[i,j] = D[j,i] = (c6 / Q[i,j])^(1/6)
    end

    H = I - ones(n,n)/n
    B = Symmetric(-0.5 * H * (D.^2) * H)

    F = eigen(B)
    idx = sortperm(F.values, rev=true)[1:dim]

    return F.vectors[:,idx] * Diagonal(sqrt.(max.(F.values[idx],0)))
end

# ═════════════════════════════════════
# RUN EXPERIMENT
# ═════════════════════════════════════
function run_all()
    L = 50

    default(
        size=(1600, 900),
        dpi=180,
        framestyle=:box,
        grid=true,
        gridalpha=0.25,
        linewidth=3,
        markersize=6,
        legendfontsize=9,
        guidefontsize=12,
        tickfontsize=10,
    )

    results_std_err = zeros(length(noise_levels), length(n_values))
    results_pol_err = zeros(length(noise_levels), length(n_values))
    results_mds_err = zeros(length(noise_levels), length(n_values))

    results_std_obj = zeros(length(noise_levels), length(n_values))
    results_pol_obj = zeros(length(noise_levels), length(n_values))

    for (ni, σ) in enumerate(noise_levels)
        println("\n===== BRUIT σ = $σ =====")

        for (ni2, n) in enumerate(n_values)

            x_init = generer_init(n, dim, L)

            # Q exact
            Q = zeros(n,n)
            for i in 1:n, j in i+1:n
                d = max(norm(x_init[i,:]-x_init[j,:]), 1e-6)
                Q[i,j] = Q[j,i] = c6 / d^6
            end

            # bruit
            Qb = bruiter_Q(Q, σ)

            Xs, obj_s, _ = solve_standard(n,dim,Qb,c6,x_init)
            Xp, obj_p, _ = solve_polaire(n,dim,Qb,c6,x_init)
            Xm = solve_mds(n,dim,Qb,c6)

            results_std_err[ni, ni2] = erreur_rydberg(Xs,Q,c6)
            results_pol_err[ni, ni2] = erreur_rydberg(Xp,Q,c6)
            results_mds_err[ni, ni2] = erreur_rydberg(Xm,Q,c6)

            results_std_obj[ni, ni2] = obj_s
            results_pol_obj[ni, ni2] = obj_p

            println(@sprintf(
                "n=%d | obj_standard=%.6e | obj_polaire=%.6e | err_standard=%.6e | err_polaire=%.6e | err_mds=%.6e",
                n, obj_s, obj_p,
                results_std_err[ni, ni2],
                results_pol_err[ni, ni2],
                results_mds_err[ni, ni2]
            ))
        end
    end

    # ═════════════════════════════════════
    # PLOTS COURBES
    # ═════════════════════════════════════
    palette_sigma = [:royalblue, :darkorange, :seagreen, :crimson]
    line_styles = [:solid, :dash, :dot, :dashdot]

    p_obj_std = plot(
        title="Objectif Ipopt — Standard",
        xlabel="n",
        ylabel="objective_value",
        yscale=:log10,
        legend=:outerright,
        palette=palette_sigma,
        size=(800, 400),
    )
    for (i, σ) in enumerate(noise_levels)
        plot!(p_obj_std, n_values, results_std_obj[i, :],
            marker=:circle, linestyle=line_styles[i], label="σ=$(σ)")
    end

    p_obj_pol = plot(
        title="Objectif Ipopt — Polaire",
        xlabel="n",
        ylabel="objective_value",
        yscale=:log10,
        legend=:outerright,
        palette=palette_sigma,
        size=(800, 400),
    )
    for (i, σ) in enumerate(noise_levels)
        plot!(p_obj_pol, n_values, results_pol_obj[i, :],
            marker=:diamond, linestyle=line_styles[i], label="σ=$(σ)")
    end

    p_err_std = plot(
        title="Erreur de reconstruction — Standard",
        xlabel="n",
        ylabel="erreur",
        yscale=:log10,
        legend=:outerright,
        palette=palette_sigma,
        size=(800, 400),
    )
    for (i, σ) in enumerate(noise_levels)
        plot!(p_err_std, n_values, results_std_err[i, :],
            marker=:circle, linestyle=line_styles[i], label="σ=$(σ)")
    end

    p_err_pol = plot(
        title="Erreur de reconstruction — Polaire",
        xlabel="n",
        ylabel="erreur",
        yscale=:log10,
        legend=:outerright,
        palette=palette_sigma,
        size=(800, 400),
    )
    for (i, σ) in enumerate(noise_levels)
        plot!(p_err_pol, n_values, results_pol_err[i, :],
            marker=:diamond, linestyle=line_styles[i], label="σ=$(σ)")
    end

    p_err_mds = plot(
        title="Erreur de reconstruction — MDS",
        xlabel="n",
        ylabel="erreur",
        yscale=:log10,
        legend=:outerright,
        palette=palette_sigma,
        size=(800, 400),
    )
    for (i, σ) in enumerate(noise_levels)
        plot!(p_err_mds, n_values, results_mds_err[i, :],
            marker=:star5, linestyle=line_styles[i], label="σ=$(σ)")
    end

    p_top = plot(p_obj_std, p_obj_pol, layout=(1,2), size=(1600, 420))
    p_bottom = plot(p_err_std, p_err_pol, p_err_mds, layout=(1,3), size=(1600, 420))

    savefig(p_top, "robustesse_bruit_objectifs.png")
    savefig(p_bottom, "robustesse_bruit_erreurs.png")
    println("\n✓ Graphes sauvegardés : robustesse_bruit_objectifs.png et robustesse_bruit_erreurs.png")

    display(p_top)
    display(p_bottom)
end

run_all()