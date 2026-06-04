using Distributed

if nprocs() == 1
    addprocs() 
end

@everywhere using JuMP, Ipopt, LinearAlgebra, StatsBase

# --- TOUT LE CALCUL EST ENFERMÉ DANS UNE FONCTION COMPILÉE ---
@everywhere function run_single_start(all_intersections, Q, n, dim, c6)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    
    # Ajout de tolérances pour accélérer la convergence d'Ipopt
    set_optimizer_attribute(model, "tol", 1e-4)
    set_optimizer_attribute(model, "max_iter", 500)
    
    @variable(model, x_opt[1:n, 1:dim] >= 0)

    # Tirage initial unique
    chosen_points = sample(all_intersections, n, replace=false)
    for i in 1:n
        for k in 1:dim
            set_start_value(x_opt[i,k], chosen_points[i][k])
        end
    end
    
    # Objectif optimisé avec sécurité anti-division par zéro (+1e-6)
    @objective(model, Min, sum(
        ( (c6 / (sum((x_opt[i,k] - x_opt[j,k])^2 for k in 1:dim) + 1e-6)^3) - Q[i,j] )^2 
        for i in 1:n for j in (i+1):n
    ))
    
    optimize!(model)
    
    # Extraction sécurisée des résultats
    try
        return (objective_value(model), value.(x_opt))
    catch
        return (Inf, nothing)
    end
end

# --- FONCTION DE GESTION DE L'INSTANCE ---
function solve_instance(n, dim, L, c6, n_starts)
    println("\n--- RUN : n = ", n, ", dim = ", dim, " ---")
    
    x = L * rand(n, dim)
    Q = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            norm_value = norm(x[i,:] - x[j,:])
            Q[i,j] = c6 / (norm_value)^6
            Q[j,i] = c6 / (norm_value)^6
        end
    end

    g = 2 * sqrt(n) |> ceil |> Int
    grid_vals = range(0, stop=L, length=g)
    println("g = ", g, " (soit ", g^dim, " intersections disponibles pour ", n, " points)")

    all_intersections = [collect(p) for p in Iterators.product(fill(grid_vals, dim)...)]

    # Distribution parallèle des lancements pré-compilés
    results = @distributed (vcat) for start in 1:n_starts
        [run_single_start(all_intersections, Q, n, dim, c6)]
    end
    
    best_obj = Inf
    best_x = nothing
    for res in results
        if res[1] < best_obj
            best_obj = res[1]
            best_x = res[2]
        end
    end
    
    println("best_objective_value = ", best_obj)
    return best_obj, best_x
end

# --- SCRIPT PRINCIPAL EXÉCUTABLE ---
const limit_points = 10
const limit_dim = 2
const number_generation_per_instance = 1
const L = 10.0
const c6 = 1.0
const n_starts = 100

for n in 10:limit_points
    for dim in 2:limit_dim
        for _ in 1:number_generation_per_instance
            solve_instance(n, dim, L, c6, n_starts)
        end
    end
end