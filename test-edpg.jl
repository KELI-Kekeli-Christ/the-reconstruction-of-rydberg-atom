using JuMP, Ipopt, LinearAlgebra, Random, Printf, Plots, Statistics

Random.seed!(42)

# ═════════════════════════════════════
# PARAMÈTRES
# ═════════════════════════════════════
n_values = [20, 30,50,100]
dim = 2
c6 = 1.0
d_min = 1e-6
n_starts = 5  # Nombre de points de départ pour multistart

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
        d = norm(x_sol[i,:] - x_sol[j,:])
        err += (c6 / d^6 - Q[i,j])^2
    end
    return err
end

# ═════════════════════════════════════
# STANDARD NLP - Single Start
# ═════════════════════════════════════
function solve_standard(n, dim, Q, c6, x_init)
    m = Model(Ipopt.Optimizer)
    set_silent(m)

    @variable(m, x[1:n,1:dim])

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end

    @expression(m, d2[i=1:n, j=i+1:n],
        sum((x[i,k]-x[j,k])^2 for k in 1:dim)
    )

    @objective(m, Min,
        sum((c6 / d2[i,j]^3 - Q[i,j])^2 for i in 1:n for j in i+1:n)
    )

    optimize!(m)
    return value.(x), termination_status(m), objective_value(m)
end

# ═════════════════════════════════════
# STANDARD NLP - MultiStart
# ═════════════════════════════════════
function solve_standard_multistart(n, dim, Q, c6, n_starts, L)
    best_obj = Inf
    best_x = zeros(n, dim)
    best_status = nothing
    results = []

    for start in 1:n_starts
        x_init = generer_init(n, dim, L)
        
        m = Model(Ipopt.Optimizer)
        set_silent(m)

        @variable(m, x[1:n,1:dim])

        for i in 1:n, d in 1:dim
            set_start_value(x[i,d], x_init[i,d])
        end

        @expression(m, d2[i=1:n, j=i+1:n],
            sum((x[i,k]-x[j,k])^2 for k in 1:dim) 
        )

        @objective(m, Min,
            sum((c6 / d2[i,j]^3 - Q[i,j])^2 for i in 1:n for j in i+1:n)
        )

        optimize!(m)
        obj_val = objective_value(m)
        
        push!(results, obj_val)
        
        if obj_val < best_obj
            best_obj = obj_val
            best_x = value.(x)
            best_status = termination_status(m)
        end
    end

    return best_x, best_status, best_obj, results
end

# ═════════════════════════════════════
# POLAIRE NLP - Single Start
# ═════════════════════════════════════
function solve_polaire(n, dim, Q, c6, x_init)
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "max_iter", 10000)
    set_optimizer_attribute(m, "tol", 1e-8)

    @variable(m, x[1:n,1:dim])
    @variable(m, R[i=1:n,j=i+1:n] >= d_min)
    @variable(m, err[i=1:n,j=i+1:n])

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
        set_lower_bound(x[i,d], 0)
        set_upper_bound(x[i,d], 100)
    end

    # Initialize R values from initial configuration
    for i in 1:n, j in i+1:n
        d = max(norm(x_init[i,:] - x_init[j,:]), d_min)
        set_start_value(R[i,j], d)
    end

    for i in 1:n, j in i+1:n
        @constraint(m,
            (x[i,1]-x[j,1])^2 + (x[i,2]-x[j,2])^2 == R[i,j]^2
        )

        @constraint(m,
            err[i,j] == (c6 / (R[i,j]^6 )) - Q[i,j]
        )
    end

    @objective(m, Min, sum(err[i,j]^2 for i in 1:n for j in i+1:n))

    optimize!(m)
    return value.(x), termination_status(m), objective_value(m)
end

# ═════════════════════════════════════
# POLAIRE NLP - MultiStart
# ═════════════════════════════════════
function solve_polaire_multistart(n, dim, Q, c6, n_starts, L)
    best_obj = Inf
    best_x = zeros(n, dim)
    best_status = nothing
    results = []

    for start in 1:n_starts
        x_init = generer_init(n, dim, L)
        
        m = Model(Ipopt.Optimizer)
        set_silent(m)
        set_optimizer_attribute(m, "max_iter", 10000)
        set_optimizer_attribute(m, "tol", 1e-8)

        @variable(m, x[1:n,1:dim])
        @variable(m, R[i=1:n,j=i+1:n] >= d_min)
        @variable(m, err[i=1:n,j=i+1:n])

        for i in 1:n, d in 1:dim
            set_start_value(x[i,d], x_init[i,d])
            set_lower_bound(x[i,d], 0)
            set_upper_bound(x[i,d], 100)
        end

        for i in 1:n, j in i+1:n
            d = max(norm(x_init[i,:] - x_init[j,:]), d_min)
            set_start_value(R[i,j], d)
        end

        for i in 1:n, j in i+1:n
            @constraint(m,
                (x[i,1]-x[j,1])^2 + (x[i,2]-x[j,2])^2 == R[i,j]^2
            )

            @constraint(m,
                err[i,j] == (c6 / (R[i,j]^6)) - Q[i,j]
            )
        end

        @objective(m, Min, sum(err[i,j]^2 for i in 1:n for j in i+1:n))

        optimize!(m)
        obj_val = objective_value(m)
        
        push!(results, obj_val)
        
        if obj_val < best_obj
            best_obj = obj_val
            best_x = value.(x)
            best_status = termination_status(m)
        end
    end

    return best_x, best_status, best_obj, results
end

# ═════════════════════════════════════
# MDS EXACT (baseline géométrique)
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

    X = F.vectors[:,idx] * Diagonal(sqrt.(max.(F.values[idx],0)))
    return X
end

# ═════════════════════════════════════
# RUN SIMULATION
# ═════════════════════════════════════
function run_all()
    L = 100

    # Single start results
    err_std_single = Float64[]
    err_pol_single = Float64[]
    err_mds = Float64[]

    # Multistart results
    err_std_multi = Float64[]
    err_pol_multi = Float64[]
    
    # Multistart variability
    std_std_multi = Float64[]
    std_pol_multi = Float64[]

    status_std = String[]
    status_pol = String[]

    for n in n_values
        x_init = generer_init(n, dim, L)

        Q = zeros(n,n)
        for i in 1:n, j in i+1:n
            d = max(norm(x_init[i,:]-x_init[j,:]), 1e-6)
            Q[i,j] = Q[j,i] = c6 / d^6
        end

        # SINGLE START
        X_std, st_std, obj_std = solve_standard(n, dim, Q, c6, generer_init(n, dim, L))
        X_pol, st_pol, obj_pol = solve_polaire(n, dim, Q, c6, generer_init(n, dim, L))
        X_mds = solve_mds(n, dim, Q, c6)

        push!(err_std_single, erreur_rydberg(X_std, Q, c6))
        push!(err_pol_single, erreur_rydberg(X_pol, Q, c6))
        push!(err_mds, erreur_rydberg(X_mds, Q, c6))

        push!(status_std, string(st_std))
        push!(status_pol, string(st_pol))

        # MULTISTART
        X_std_ms, st_std_ms, obj_std_ms, results_std_ms = solve_standard_multistart(n, dim, Q, c6, n_starts, L)
        X_pol_ms, st_pol_ms, obj_pol_ms, results_pol_ms = solve_polaire_multistart(n, dim, Q, c6, n_starts, L)

        push!(err_std_multi, erreur_rydberg(X_std_ms, Q, c6))
        push!(err_pol_multi, erreur_rydberg(X_pol_ms, Q, c6))
        
        push!(std_std_multi, std(results_std_ms))
        push!(std_pol_multi, std(results_pol_ms))

        @printf("\n✓ n=%d complété (single + multistart)\n", n)
    end

    # ═════════════════════════════════════
    # PLOT 1: Comparaison Single vs Multi
    # ═════════════════════════════════════
    eps_plot = 1e-14
    err_std_single_plot = err_std_single .+ eps_plot
    err_pol_single_plot = err_pol_single .+ eps_plot
    err_mds_plot = err_mds .+ eps_plot
    err_std_multi_plot = err_std_multi .+ eps_plot
    err_pol_multi_plot = err_pol_multi .+ eps_plot

    p1 = plot(
        n_values, err_std_single_plot,
        label="Standard (Single Start)",
        marker=:circle,
        yscale=:log10,
        linewidth=2,
        legend=:topleft,
        size=(900, 600),
        palette=:Set1_9
    )

    plot!(p1, n_values, err_std_multi_plot,
        label="Standard (MultiStart)",
        marker=:star5,
        linewidth=2,
        linestyle=:dash
    )

    plot!(p1, n_values, err_pol_single_plot,
        label="Polaire (Single Start)",
        marker=:square,
        linewidth=2
    )

    plot!(p1, n_values, err_pol_multi_plot,
        label="Polaire (MultiStart)",
        marker=:diamond,
        linewidth=2,
        linestyle=:dash
    )

    plot!(p1, n_values, err_mds_plot,
        label="MDS exact",
        marker=:cross,
        linewidth=2,
        color=:gray
    )

    xlabel!("n (nombre d'atomes)")
    ylabel!("Erreur Rydberg (log10)")
    title!("Comparaison: Single Start vs MultiStart (n_starts=$n_starts)")

    savefig(p1, "comparison_singleVSmulti.png")
    @printf("\n✓ Graphe 1 sauvegardé: comparison_singleVSmulti.png\n")

    # ═════════════════════════════════════
    # PLOT 2: Impact du MultiStart (amélioration)
    # ═════════════════════════════════════
    improvement_std = (err_std_single .- err_std_multi) ./ err_std_single .* 100
    improvement_pol = (err_pol_single .- err_pol_multi) ./ err_pol_single .* 100

    p2 = plot(
        n_values, improvement_std,
        label="Standard",
        marker=:circle,
        linewidth=2.5,
        legend=:topleft,
        size=(800, 500),
        palette=:Set1_9
    )

    plot!(p2, n_values, improvement_pol,
        label="Polaire",
        marker=:square,
        linewidth=2.5
    )

    xlabel!("n (nombre d'atomes)")
    ylabel!("Amélioration (%)")
    title!("Impact du MultiStart: % réduction d'erreur")
    hline!(p2, [0], line=:dash, color=:gray, label=nothing)

    savefig(p2, "multistart_improvement.png")
    @printf("✓ Graphe 2 sauvegardé: multistart_improvement.png\n")

    # ═════════════════════════════════════
    # PLOT 3: Variabilité MultiStart (std)
    # ═════════════════════════════════════
    p3 = plot(
        n_values, std_std_multi,
        label="Standard",
        marker=:circle,
        linewidth=2.5,
        legend=:topleft,
        size=(800, 500),
        yscale=:log10,
        palette=:Set1_9
    )

    plot!(p3, n_values, std_pol_multi,
        label="Polaire",
        marker=:square,
        linewidth=2.5
    )

    xlabel!("n (nombre d'atomes)")
    ylabel!("Écart-type (log10)")
    title!("Variabilité du MultiStart across $n_starts essais")

    savefig(p3, "multistart_variability.png")
    @printf("✓ Graphe 3 sauvegardé: multistart_variability.png\n")

    # ═════════════════════════════════════
    # PLOT 4: Comparaison des 3 méthodes (BEST)
    # ═════════════════════════════════════
    p4 = plot(
        n_values, err_std_multi_plot,
        label="Standard (MultiStart)",
        marker=:circle,
        linewidth=3,
        markersize=8,
        yscale=:log10,
        legend=:topleft,
        size=(1000, 650),
        palette=:Set1_9
    )

    plot!(p4, n_values, err_pol_multi_plot,
        label="Polaire (MultiStart)",
        marker=:square,
        linewidth=3,
        markersize=8
    )

    plot!(p4, n_values, err_mds_plot,
        label="MDS exact",
        marker=:star5,
        linewidth=3,
        markersize=10,
        color=:green
    )

    xlabel!("n (nombre d'atomes)", fontsize=12)
    ylabel!("Erreur Rydberg (log10)", fontsize=12)
    title!("Comparaison des 3 Méthodes - Meilleure Solution par Taille", fontsize=13)
    
    savefig(p4, "comparison_3methods.png")
    @printf("✓ Graphe 4 sauvegardé: comparison_3methods.png\n")

    # ═════════════════════════════════════
    # RÉSUMÉ TEXTUEL
    # ═════════════════════════════════════
    println("\n" * "="^60)
    println("RÉSULTATS DÉTAILLÉS - SINGLE START vs MULTISTART")
    println("="^60)
    
    for i in 1:length(n_values)
        n = n_values[i]
        println("\n" * "─"^60)
        println("n = $n atomes")
        println("─"^60)
        
        println("STANDARD:")
        @printf("  Single:  erreur=%.6e | status=%s\n", err_std_single[i], status_std[i])
        @printf("  Multi:   erreur=%.6e | amélioration: %+.2f%%\n", 
            err_std_multi[i], improvement_std[i])
        @printf("  Variabilité: std=%.6e\n", std_std_multi[i])
        
        println("\nPOLAIRE:")
        @printf("  Single:  erreur=%.6e | status=%s\n", err_pol_single[i], status_pol[i])
        @printf("  Multi:   erreur=%.6e | amélioration: %+.2f%%\n", 
            err_pol_multi[i], improvement_pol[i])
        @printf("  Variabilité: std=%.6e\n", std_pol_multi[i])
        
        println("\nMDS (baseline):")
        @printf("  Erreur: %.6e\n", err_mds[i])
    end
    
    println("\n" * "="^60)
    println("CLASSEMENT FINAL (Meilleure Solution)")
    println("="^60)
    
    for i in 1:length(n_values)
        n = n_values[i]
        results = [
            ("Standard (MultiStart)", err_std_multi[i]),
            ("Polaire (MultiStart)", err_pol_multi[i]),
            ("MDS exact", err_mds[i])
        ]
        sorted = sort(results, by=x->x[2])
        
        println("\nn = $n atomes:")
        for (rank, (method, err)) in enumerate(sorted)
            rank_symbol = rank == 1 ? "🏆" : rank == 2 ? "🥈" : "🥉"
            @printf("  %s #%d: %-25s → erreur = %.6e\n", rank_symbol, rank, method, err)
        end
    end
    
    println("\n" * "="^60)
end

run_all()