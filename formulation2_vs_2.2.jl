using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

n_values  = [2, 3, 10,20]
limit_dim = 2           # Dimension fixe : 2D
L         = 10
c6        = 1.0

const MOI = JuMP.MOI

results = DataFrame(
    n           = Int[],
    dim         = Int[],
    formulation = String[],
    obj         = Float64[],
    status      = String[],
    time_s      = Float64[]
)

# --- Génère une initialisation aléatoire, strictement positive, sans doublons ---
function generer_init(n, dim, L; seuil=1e-3)
    x = L * rand(n, dim)
    for i in 1:n
        # Sécurité 1 : aucun point à l'origine
        while norm(x[i, :]) < seuil
            x[i, :] = L * rand(dim)
        end
        # Sécurité 2 : aucune superposition entre atomes
        for j in 1:(i-1)
            tentatives = 0
            while norm(x[i, :] - x[j, :]) < seuil
                x[i, :] .+= seuil * rand(dim)
                x[i, :]   = clamp.(x[i, :], 0.0, L)
                tentatives += 1
                if tentatives > 100
                    x[i, :] = L * rand(dim)
                    tentatives = 0
                end
            end
        end
    end
    return x
end

# CORRECTION BUG 4 : suppression de la boucle `for dim in 2:limit_dim`
# qui ne tournait qu'une seule fois (dim=2). On fixe dim = limit_dim directement.
dim = limit_dim

for n in n_values
    println("\n=== n=$n, dim=$dim ===")

    # CORRECTION BUG 1 : x_vrais et x_init sont deux tirages DISTINCTS
    # x_vrais → configuration cible pour calculer Q (le "vrai" problème)
    # x_init  → point de départ de l'optimiseur (warm-start, différent de x_vrais)
    x_vrais = generer_init(n, dim, L)   # config cible
    x_init  = generer_init(n, dim, L)   # warm-start (aléatoire, non nul, unique)

    # Matrice cible Q calculée depuis x_vrais uniquement
    Q = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            d = norm(x_vrais[i, :] - x_vrais[j, :])
            if d < 1e-5; d = 1e-5; end
            Q[i, j] = c6 / d^6
            Q[j, i] = Q[i, j]
        end
    end

    # ================================================================
    # FORMULATION STANDARD
    # min  sum_{i<j} ( c6/||xi-xj||^6 - Q[i,j] )^2
    # ================================================================
    t0 = time()
    m1 = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m1, "print_level", 0)
    set_optimizer_attribute(m1, "tol", 1e-12)

    @variable(m1, x1[1:n, 1:dim] >= 0)
    for i in 1:n, d in 1:dim
        set_start_value(x1[i, d], x_init[i, d])
    end

    @NLobjective(m1, Min, sum(
        ( c6 / (sum((x1[i,k] - x1[j,k])^2 for k in 1:dim))^3 - Q[i,j] )^2
        for i in 1:n for j in i+1:n
    ))

    optimize!(m1)
    t1 = time() - t0
    obj1 = objective_value(m1)
    stat1 = string(termination_status(m1))
    println("  [Standard]   status=$stat1  obj=$obj1  t=$(round(t1, digits=3))s")
    push!(results, (n, dim, "standard", obj1, stat1, t1))

    # ================================================================
    # FORMULATION REFORMULÉE avec variables auxiliaires
    #   z[i,j] = ||xi - xj||²            (distance²)
    #   u[i,j] = z[i,j]²                 (distance⁴)
    #   w[i,j] = u[i,j] * z[i,j]         (distance⁶)
    #   r[i,j] * w[i,j] = c6             (r = c6/d^6)
    # Objectif : min sum_{i<j} ( r[i,j] - Q[i,j] )^2
    # ================================================================
    t0 = time()
    m2 = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m2, "print_level", 0)
    set_optimizer_attribute(m2, "tol", 1e-12)

    @variable(m2, x2[1:n, 1:dim] >= 0)
    @variable(m2, r[i=1:n, j=(i+1):n])   # c6 / d^6
    @variable(m2, w[i=1:n, j=(i+1):n])   # d^6
    @variable(m2, u[i=1:n, j=(i+1):n])   # d^4
    @variable(m2, z[i=1:n, j=(i+1):n])   # d^2

    # Warm-start depuis x_init (le vrai warm-start, distinct de x_vrais)
    for i in 1:n, d in 1:dim
        set_start_value(x2[i, d], x_init[i, d])
    end
    for i in 1:n
        for j in i+1:n
            d_sq  = max(sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim), 1e-10)
            z_val = d_sq
            u_val = d_sq^2
            w_val = d_sq^3
            set_start_value(z[i, j], z_val)
            set_start_value(u[i, j], u_val)
            set_start_value(w[i, j], w_val)
            set_start_value(r[i, j], c6 / w_val)
        end
    end

    # CORRECTION BUG 3 : toutes les contraintes non-linéaires → @NLconstraint
    # @constraint ne gère que le linéaire/quadratique.
    # w = u*z est cubique, r*w = c6 est bilinéaire → @NLconstraint obligatoire.
    for i in 1:n
        for j in i+1:n
            # Contrainte 1 (bilinéaire) : r * w = c6
            @NLconstraint(m2, r[i,j] * w[i,j] == c6)
            # Contrainte 2 (quadratique) : z = ||xi - xj||²
            @NLconstraint(m2, z[i,j] == sum((x2[i,k] - x2[j,k])^2 for k in 1:dim))
            # Contrainte 3 (cubique) : w = u * z
            @NLconstraint(m2, w[i,j] == u[i,j] * z[i,j])
            # Contrainte 4 (quadratique) : u = z²
            @NLconstraint(m2, u[i,j] == z[i,j]^2)
        end
    end

    @objective(m2, Min, sum((r[i,j] - Q[i,j])^2 for i in 1:n for j in i+1:n))

    optimize!(m2)
    t2 = time() - t0
    obj2 = objective_value(m2)
    stat2 = string(termination_status(m2))
    println("  [Reformulée] status=$stat2  obj=$obj2  t=$(round(t2, digits=3))s")
    push!(results, (n, dim, "reformulee", obj2, stat2, t2))
end

csv_path = joinpath(@__DIR__, "results_formulation_vs.csv")
CSV.write(csv_path, results)
println("\nRésultats sauvegardés dans : $csv_path")
println(results)