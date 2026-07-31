# ═══════════════════════════════════════════════════════════════
#  Formulations NLP avec L variable dans l'objectif :
#      min  f(x) + α · L
#  sous  0 ≤ x_{ik} ≤ L.
#
#  Mot-clé `zmin` :
#    - Float : contrainte de non-collision ‖x_i-x_j‖ ≥ zmin
#              (zmin = (c6/Qmax)^{1/6} recommandé)
#    - nothing : pas de contrainte de distance minimale (exp. 2)
#
#  Retour : (status, obj, x, L_opt) ou nothing
# ═══════════════════════════════════════════════════════════════

function _ipopt_model()
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    set_optimizer_attribute(m, "tol", 1e-12)
    set_optimizer_attribute(m, "max_iter", 10000)
    return m
end

function _setup_L_and_x(m, n, dim, x_init; L_init::Float64, L_lb::Float64)
    @variable(m, L >= L_lb)
    set_start_value(L, max(L_init, L_lb))
    @variable(m, x[1:n, 1:dim] >= 0)
    for i in 1:n, d in 1:dim
        @constraint(m, x[i, d] <= L)
        set_start_value(x[i, d], clamp(x_init[i, d], 0.0, max(L_init, L_lb)))
    end
    return L, x
end

function _ok_status(status)
    status in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED,
               MOI.ITERATION_LIMIT, MOI.SLOW_PROGRESS)
end

"""Standard : Σ (c6/z³ - Q)² + α L,  z = ‖x_i-x_j‖²."""
function solve_standard_Lvar(n, dim, Q, c6, x_init;
                             alpha::Float64=1.0,
                             zmin=nothing,
                             L_init::Float64=10.0,
                             L_lb::Float64=1e-3)
    m = _ipopt_model()
    edges = active_pairs(Q)
    L, x = _setup_L_and_x(m, n, dim, x_init; L_init=L_init, L_lb=L_lb)

    @expression(m, d2[i=1:n, j=i+1:n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim))

    if zmin !== nothing
        zmin2 = zmin^2
        for i in 1:n, j in i+1:n
            @constraint(m, d2[i, j] >= zmin2)
        end
    end

    @objective(m, Min,
        sum((c6 / d2[i, j]^3 - Q[i, j])^2 for (i, j) in edges) + alpha * L
    )

    optimize!(m)
    status = termination_status(m)
    _ok_status(status) || return nothing
    return (status, objective_value(m), value.(x), value(L))
end

"""Simple : Σ (r - d)² + α L."""
function solve_simple_Lvar(n, dim, Q, c6, x_init;
                           alpha::Float64=1.0,
                           zmin=nothing,
                           L_init::Float64=10.0,
                           L_lb::Float64=1e-3)
    m = _ipopt_model()
    edges = active_pairs(Q)
    L, x = _setup_L_and_x(m, n, dim, x_init; L_init=L_init, L_lb=L_lb)

    r_lb = zmin === nothing ? 1e-8 : Float64(zmin)
    @variable(m, r[i=1:n, j=i+1:n] >= r_lb)
    @variable(m, d[i=1:n, j=i+1:n])

    for (i, j) in edges
        set_start_value(r[i, j], max(norm(x_init[i, :] - x_init[j, :]), r_lb))
        set_start_value(d[i, j], (c6 / Q[i, j])^(1 / 6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim))

    for (i, j) in edges
        @constraint(m, r[i, j] == sqrt(sq_dist[i, j]))
        @constraint(m, d[i, j] == (c6 / Q[i, j])^(1 / 6))
        if zmin !== nothing
            @constraint(m, sq_dist[i, j] >= zmin^2)
        end
    end

    @objective(m, Min,
        sum((r[i, j] - d[i, j])^2 for (i, j) in edges) + alpha * L
    )

    optimize!(m)
    status = termination_status(m)
    _ok_status(status) || return nothing
    return (status, objective_value(m), value.(x), value(L))
end

"""Simple square : Σ (r² - d²)² + α L."""
function solve_simple_square_Lvar(n, dim, Q, c6, x_init;
                                  alpha::Float64=1.0,
                                  zmin=nothing,
                                  L_init::Float64=10.0,
                                  L_lb::Float64=1e-3)
    m = _ipopt_model()
    edges = active_pairs(Q)
    L, x = _setup_L_and_x(m, n, dim, x_init; L_init=L_init, L_lb=L_lb)

    r_lb = zmin === nothing ? 1e-8 : Float64(zmin)
    @variable(m, r[i=1:n, j=i+1:n] >= r_lb)
    @variable(m, d[i=1:n, j=i+1:n])

    for (i, j) in edges
        set_start_value(r[i, j], max(norm(x_init[i, :] - x_init[j, :]), r_lb))
        set_start_value(d[i, j], (c6 / Q[i, j])^(1 / 6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim))

    for (i, j) in edges
        @constraint(m, r[i, j]^2 == sq_dist[i, j])
        @constraint(m, d[i, j] == (c6 / Q[i, j])^(1 / 6))
        if zmin !== nothing
            @constraint(m, sq_dist[i, j] >= zmin^2)
        end
    end

    @objective(m, Min,
        sum((r[i, j]^2 - d[i, j]^2)^2 for (i, j) in edges) + alpha * L
    )

    optimize!(m)
    status = termination_status(m)
    _ok_status(status) || return nothing
    return (status, objective_value(m), value.(x), value(L))
end

"""Taylor : Σ (36/d¹⁴)(r - d)² + α L."""
function solve_taylor_Lvar(n, dim, Q, c6, x_init;
                           alpha::Float64=1.0,
                           zmin=nothing,
                           L_init::Float64=10.0,
                           L_lb::Float64=1e-3)
    m = _ipopt_model()
    edges = active_pairs(Q)
    L, x = _setup_L_and_x(m, n, dim, x_init; L_init=L_init, L_lb=L_lb)

    r_lb = zmin === nothing ? 1e-8 : Float64(zmin)
    @variable(m, r[i=1:n, j=i+1:n] >= r_lb)
    @variable(m, d[i=1:n, j=i+1:n])

    for (i, j) in edges
        set_start_value(r[i, j], max(norm(x_init[i, :] - x_init[j, :]), r_lb))
        set_start_value(d[i, j], (c6 / Q[i, j])^(1 / 6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim))

    for (i, j) in edges
        @constraint(m, r[i, j] == sqrt(sq_dist[i, j]))
        @constraint(m, d[i, j] == (c6 / Q[i, j])^(1 / 6))
        if zmin !== nothing
            @constraint(m, sq_dist[i, j] >= zmin^2)
        end
    end

    @objective(m, Min,
        sum((36 / d[i, j]^15) * (r[i, j] - d[i, j])^2 for (i, j) in edges) + alpha * L
    )

    optimize!(m)
    status = termination_status(m)
    _ok_status(status) || return nothing
    return (status, objective_value(m), value.(x), value(L))
end

"""Taylor square : Σ (36/d¹⁴)(r² - d²)² + α L."""
function solve_taylor_square_Lvar(n, dim, Q, c6, x_init;
                                  alpha::Float64=1.0,
                                  zmin=nothing,
                                  L_init::Float64=10.0,
                                  L_lb::Float64=1e-3)
    m = _ipopt_model()
    edges = active_pairs(Q)
    L, x = _setup_L_and_x(m, n, dim, x_init; L_init=L_init, L_lb=L_lb)

    r_lb = zmin === nothing ? 1e-8 : Float64(zmin)
    @variable(m, r[i=1:n, j=i+1:n] >= r_lb)
    @variable(m, d[i=1:n, j=i+1:n])

    for (i, j) in edges
        set_start_value(r[i, j], max(norm(x_init[i, :] - x_init[j, :]), r_lb))
        set_start_value(d[i, j], (c6 / Q[i, j])^(1 / 6))
    end

    @expression(m, sq_dist[i=1:n, j=(i+1):n],
        sum((x[i, k] - x[j, k])^2 for k in 1:dim))

    for (i, j) in edges
        @constraint(m, r[i, j]^2 == sq_dist[i, j])
        @constraint(m, d[i, j] == (c6 / Q[i, j])^(1 / 6))
        if zmin !== nothing
            @constraint(m, sq_dist[i, j] >= zmin^2)
        end
    end

    @objective(m, Min,
        sum((36 / d[i, j]^15) * (r[i, j]^2 - d[i, j]^2)^2 for (i, j) in edges) + alpha * L
    )

    optimize!(m)
    status = termination_status(m)
    _ok_status(status) || return nothing
    return (status, objective_value(m), value.(x), value(L))
end

const LVAR_FORMULATIONS = Pair{String, Function}[
    "Standard"      => solve_standard_Lvar,
    "Simple"        => solve_simple_Lvar,
    "Simple_square" => solve_simple_square_Lvar,
    "Taylor"        => solve_taylor_Lvar,
    "Taylor_square" => solve_taylor_square_Lvar,
]
