# ═══════════════════════════════════════════════════════════════
#  §3  FORMULATION 1 — STANDARD
#  Positions libres dans R² (invariance par translation/rotation)
# ═══════════════════════════════════════════════════════════════
function solve_standard(n, dim, Q, c6, L, x_init,z_min=0.0016)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0<=x[1:n, 1:dim]<=L)          
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
        sum((c6 / d2[i, j]^3 - Q[i, j])^2 for (i, j) in edges)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end

#= [désactivé — hors formulations retenues]
function solve_new_standard(n,dim,Q,c6,L,x_init,z_min=0.0016)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

   r = zeros(n, n)
    for (i, j) in edges
        r[i, j] = (c6 / Q[i, j])^(1/6) 
    end

    @variable(m, 0 <= x[1:n, 1:dim] <= L)          
    
    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n], sum((x[i, d] - x[j, d])^2 for d in 1:dim))
    
    @expression(m, dist[i=1:n, j=(i+1):n], sqrt(sq_dist[i, j]))

    @objective(m, Min, sum((dist[i, j] - r[i, j])^2 for (i, j) in edges))
    
    for i in 1:n, j in (i+1):n
        @constraint(m, sq_dist[i, j] >= z_min^2)
    end

    optimize!(m)
    status = termination_status(m)
    
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    else
        return (status, nothing, nothing) # Handle failure gracefully
    end
end
=#

# ═══════════════════════════════════════════════════════════════
#  §4  FORMULATION 2 — REFORMULÉE
#= [désactivé — hors formulations retenues]
# ═══════════════════════════════════════════════════════════════
function solve_reformulee(n, dim, Q, c6, L, x_init,z_min=0.0016)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level",        0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0<=x2[1:n, 1:dim]<=L)         


    @variable(m, z[i=1:n, j=(i+1):n])
    @variable(m, u[i=1:n, j=(i+1):n])
    @variable(m, w[i=1:n, j=(i+1):n])
    @variable(m, r[i=1:n, j=(i+1):n])

    for i in 1:n, d in 1:dim
        set_start_value(x2[i, d], x_init[i, d])
    end
    for (i, j) in edges
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
        sum((r[i,j]-Q[i,j])^2 for (i, j) in edges)
    )

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x2))
    end
end
=#

# ═══════════════════════════════════════════════════════════════
#  §5  FORMULATION 3 — POLAIRE
#= [désactivé — hors formulations retenues]
# ═══════════════════════════════════════════════════════════════
function solve_polaire(n, dim, Q, c6, L, x_init,z_min=0.0016,d_min=0.04)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",        0)
    set_optimizer_attribute(model, "tol",                1e-12)
    set_optimizer_attribute(model, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(model, 0<=x[1:n, 1:dim]<=L)     
    @variable(model, -pi <= theta[i=1:n, j=i+1:n] <= pi)
    @variable(model, R[i=1:n, j=i+1:n] >= d_min)

    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end
    for (i, j) in edges
        dx = x_init[i,1] - x_init[j,1]
        dy = x_init[i,2] - x_init[j,2]
        set_start_value(theta[i,j], atan(dy,dx))
        set_start_value(R[i,j], norm([dx,dy]))
    end

    @variable(model, error_pair[i=1:n, j=i+1:n])
    for i in 1:n, j in i+1:n
        @constraint(model, x[i,1]-x[j,1] == R[i,j]*cos(theta[i,j]))
        @constraint(model, x[i,2]-x[j,2] == R[i,j]*sin(theta[i,j]))
        @constraint(model, R[i,j] >= z_min)
        @constraint(model, error_pair[i,j] == (c6/R[i,j]^6) - Q[i,j])
    end
    @objective(model, Min,
        sum(error_pair[i,j]^2 for (i, j) in edges)
    )

    optimize!(model)
    status = termination_status(model)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(model), value.(x))
    end
    return nothing
end
=#

# ═══════════════════════════════════════════════════════════════
#  §5b  FORMULATION — SUROGATE
#= [désactivé — hors formulations retenues]
# ═══════════════════════════════════════════════════════════════
function solve_surogate(n, dim, Q, c6, L, x_init,z_min=0.0016)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n])
    @variable(m, w[i=1:n, j=i+1:n])
    @variable(m, z[i=1:n, j=i+1:n])
    @variable(m, mu[i=1:n, j=i+1:n])

    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end
    for (i, j) in edges
        d_sq = sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim)
        set_start_value(z[i, j], d_sq)
        set_start_value(mu[i, j], d_sq^2)
        set_start_value(w[i, j], d_sq^3)
        set_start_value(r[i, j], c6 / d_sq^3)
    end

    for i in 1:n, j in i+1:n
        @constraint(m, z[i,j] >= z_min)
    end

    @objective(m, Min, sum((r[i, j] - Q[i, j])^2 for (i, j) in edges))

    for i in 1:n, j in i+1:n
        @constraint(m,
            r[i,j] * w[i,j] + z[i,j] + w[i,j] + mu[i,j] - mu[i,j] * z[i,j] - z[i,j]^2
            - sum((x[i,k] - x[j,k])^2 for k in 1:dim) - c6 == 0
        )
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
    return nothing
end
=#

# ═══════════════════════════════════════════════════════════════
#  §5c  FORMULATION — QUADRATIQUE (matrices explicites)
#= [désactivé — hors formulations retenues]
# ═══════════════════════════════════════════════════════════════
function solve_quadratique(n, dim, Q, c6, L, x_init, z_min=0.0016)
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n])
    @variable(m, w[i=1:n, j=i+1:n])
    @variable(m, z[i=1:n, j=i+1:n])
    @variable(m, mu[i=1:n, j=i+1:n])

    for i in 1:n, d in 1:dim
        set_start_value(x[i, d], x_init[i, d])
    end
    for (i, j) in edges
        d_sq = sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim)
        set_start_value(z[i, j], d_sq)
        set_start_value(mu[i, j], d_sq^2)
        set_start_value(w[i, j], d_sq^3)
        set_start_value(r[i, j], c6 / d_sq^3)
    end

    @objective(m, Min, sum((r[i, j] - Q[i, j])^2 for (i, j) in edges))

    for i in 1:n, j in i+1:n
        X_vec = [r[i,j], w[i,j], mu[i,j], z[i,j], x[i,1], x[i,2], x[j,1], x[j,2]]

        Q1 = zeros(8, 8); Q1[1,2] = 1; Q1[2,1] = 1
        @constraint(m, 0.5 * X_vec' * Q1 * X_vec == c6)

        Q2 = zeros(8, 8); Q2[3,4] = 1; Q2[4,3] = 1
        b2 = zeros(8); b2[2] = -1
        @constraint(m, 0.5 * X_vec' * Q2 * X_vec + b2' * X_vec == 0)

        Q3 = zeros(8, 8); Q3[4,4] = 2
        b3 = zeros(8); b3[3] = -1
        @constraint(m, 0.5 * X_vec' * Q3 * X_vec + b3' * X_vec == 0)

        Q4 = zeros(8, 8)
        Q4[5,5] = 2; Q4[6,6] = 2
        Q4[7,7] = 2; Q4[8,8] = 2
        Q4[5,7] = -2; Q4[7,5] = -2
        Q4[6,8] = -2; Q4[8,6] = -2
        b4 = zeros(8); b4[4] = -1
        @constraint(m, 0.5 * X_vec' * Q4 * X_vec + b4' * X_vec == 0)
        @constraint(m, z[i,j] >= z_min)
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
    return nothing
end
=#

# ═══════════════════════════════════════════════════════════════
#  §6  FORMULATION 4 — MDS EXACT (solution algébrique globale)
#
#  Q_ij = c6/d_ij^6  →  d_ij = (c6/Q_ij)^(1/6)  exactement connu
#  → Positionnement par distances via double centrage + eigendecomposition
#  → Erreur Rydberg ≈ 0 garanti si Q parfait, O(n³), sans itération
# ═══════════════════════════════════════════════════════════════
function solve_mds(n, dim, Q, c6)
    # Étape 1 : distances cibles
    edges = active_pairs(Q)
    D = zeros(Float64, n, n)
    dvals = Float64[]
    for (i, j) in edges
        d_ij = (c6 / Q[i,j])^(1/6)
        D[i,j] = d_ij;  D[j,i] = d_ij
        push!(dvals, d_ij)
    end

    isempty(edges) && return nothing

    fallback_d = maximum(dvals)
    for i in 1:n, j in i+1:n
        if D[i,j] == 0.0
            D[i,j] = fallback_d
            D[j,i] = fallback_d
        end
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

function formulation_taylor(Q,n,dim,L,c6,x_init,z_min=0.0016,d_min=0.04)

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n] >= d_min)
    @variable(m, d[i=1:n, j=i+1:n])

    for i in 1:n, k in 1:dim
        set_start_value(x[i, k], x_init[i, k])
    end
    for (i, j) in edges
        set_start_value(r[i, j], norm(x_init[i,:] - x_init[j,:]))
        set_start_value(d[i, j], (c6 / Q[i,j])^(1/6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n], sum((x[i, k] - x[j, k])^2 for k in 1:dim))
    @objective(m, Min, sum((36/d[i, j]^14) * (r[i, j] - d[i, j])^2 for (i, j) in edges))

    for (i, j) in edges
        @constraint(m, sq_dist[i, j] >= z_min^2)
        @constraint(m, r[i,j] == sqrt(sq_dist[i, j]))
        @constraint(m, d[i,j] == (c6 / Q[i,j])^(1/6))
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end


function formulation_taylor_square(Q,n,dim,L,c6,x_init,z_min=0.0016,d_min=0.04)

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n] >= d_min)
    @variable(m, d[i=1:n, j=i+1:n])

    for i in 1:n, k in 1:dim
        set_start_value(x[i, k], x_init[i, k])
    end
    for (i, j) in edges
        set_start_value(r[i, j], norm(x_init[i,:] - x_init[j,:]))
        set_start_value(d[i, j], (c6 / Q[i,j])^(1/6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n], sum((x[i, k] - x[j, k])^2 for k in 1:dim))
    @objective(m, Min, sum((36/d[i, j]^14) * (r[i, j]^2 - d[i, j]^2)^2 for (i, j) in edges))

    for (i, j) in edges
        @constraint(m, sq_dist[i, j] >= z_min^2)
        @constraint(m, r[i,j]^2 == sq_dist[i, j])
        @constraint(m, d[i,j] == (c6 / Q[i,j])^(1/6))
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end


function formulation_simple_square(Q,n,dim,L,c6,x_init,z_min=0.0016,d_min=0.04)

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n] >= d_min)
    @variable(m, d[i=1:n, j=i+1:n])

    for i in 1:n, k in 1:dim
        set_start_value(x[i, k], x_init[i, k])
    end
    for (i, j) in edges
        set_start_value(r[i, j], norm(x_init[i,:] - x_init[j,:]))
        set_start_value(d[i, j], (c6 / Q[i,j])^(1/6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n], sum((x[i, k] - x[j, k])^2 for k in 1:dim))
    @objective(m, Min, sum((r[i, j]^2 - d[i, j]^2)^2 for (i, j) in edges))

    for (i, j) in edges
        @constraint(m, sq_dist[i, j] >= z_min^2)
        @constraint(m, r[i,j]^2 == sq_dist[i, j])
        @constraint(m, d[i,j] == (c6 / Q[i,j])^(1/6))
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end


function formulation_simple(Q,n,dim,L,c6,x_init,z_min=0.04)

    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol",                1e-12)
    set_optimizer_attribute(m, "max_iter",           10000)
    edges = active_pairs(Q)

    @variable(m, 0 <= x[1:n, 1:dim] <= L)
    @variable(m, r[i=1:n, j=i+1:n] >= d_min)
    @variable(m, d[i=1:n, j=i+1:n])

    for i in 1:n, k in 1:dim
        set_start_value(x[i, k], x_init[i, k])
    end
    for (i, j) in edges
        set_start_value(r[i, j], norm(x_init[i,:] - x_init[j,:]))
        set_start_value(d[i, j], (c6 / Q[i,j])^(1/6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n], sum((x[i, k] - x[j, k])^2 for k in 1:dim))
    @objective(m, Min, sum((r[i, j] - d[i, j])^2 for (i, j) in edges))

    for (i, j) in edges
        @constraint(m, sq_dist[i, j] >= z_min^2)
        @constraint(m, r[i,j] == sqrt(sq_dist[i, j]))
        @constraint(m, d[i,j] == (c6 / Q[i,j])^(1/6))
    end

    optimize!(m)
    status = termination_status(m)
    if status in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS]
        return (status, objective_value(m), value.(x))
    end
end

solve_taylor(n, dim, Q, c6, L, x_init, z_min)        = formulation_taylor(Q, n, dim, L, c6, x_init, z_min)
solve_taylor_square(n, dim, Q, c6, L, x_init, z_min) = formulation_taylor_square(Q, n, dim, L, c6, x_init, z_min)
solve_simple_square(n, dim, Q, c6, L, x_init, z_min) = formulation_simple_square(Q, n, dim, L, c6, x_init, z_min)
solve_simple(n, dim, Q, c6, L, x_init, z_min)        = formulation_simple(Q, n, dim, L, c6, x_init, z_min)