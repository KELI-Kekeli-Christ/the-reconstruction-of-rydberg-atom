using JuMP
using Ipopt
using LinearAlgebra
using DataFrames
using Random

Random.seed!(42)

# --- CONFIGURATION ---
n_values  = [2, 3, 10, 50, 100]
limit_dim = 2
L         = 100
c6        = 1.0

# Nombre de restarts selon la taille du problème
# Plus n est grand, plus le paysage est accidenté → plus de tentatives
function nb_restarts(n)
    if n <= 3;   return 3
    elseif n <= 10;  return 8
    elseif n <= 50;  return 15
    else;            return 20
    end
end

results = DataFrame(
    n                   = Int[],
    formulation         = String[],
    obj                 = Float64[],
    status              = String[],
    erreur_rydberg      = Float64[],
    erreur_polaire_max  = Float64[],
    n_restarts_used     = Int[],
    time_s              = Float64[]
)

function rydberg(x_opt, Q, c6, n, dim)
    return sum(( c6 / (sum((x_opt[i,k] - x_opt[j,k])^2 for k in 1:dim))^3 - Q[i,j] )^2
                for i in 1:n for j in i+1:n)
end

function verif_coherence_polaire(x_sol, R_sol, theta_sol, n)
    erreur_max = 0.0
    for i in 1:n
        for j in i+1:n
            err_x = (x_sol[i, 1] - x_sol[j, 1]) - R_sol[i, j] * cos(theta_sol[i, j])
            err_y = (x_sol[i, 2] - x_sol[j, 2]) - R_sol[i, j] * sin(theta_sol[i, j])
            erreur_max = max(erreur_max, abs(err_x), abs(err_y))
        end
    end
    return erreur_max
end

# --- Génère une initialisation valide (strictement positive, sans superposition) ---
function generer_init(n, L, limit_dim; seuil=1e-3)
    x = L * rand(n, limit_dim)
    for i in 1:n
        while norm(x[i, :]) < seuil
            x[i, :] = L * rand(limit_dim)
        end
        for j in 1:(i-1)
            tentatives = 0
            while norm(x[i, :] - x[j, :]) < seuil
                x[i, :] .+= seuil * rand(limit_dim)
                tentatives += 1
                if tentatives > 100   # garde-fou anti-boucle infinie
                    x[i, :] = L * rand(limit_dim)
                    tentatives = 0
                end
            end
        end
    end
    return x
end

# --- Résolution d'une instance avec une initialisation donnée ---
# Retourne (statut, obj, x_sol, R_sol, theta_sol) ou rien si échec
function resoudre_instance(n, Q, c6, L, limit_dim, x_init; verbose=false)

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level",  verbose ? 5 : 0)
    set_optimizer_attribute(model, "tol",          1e-12)
    set_optimizer_attribute(model, "max_iter",     3000)
    set_optimizer_attribute(model, "bound_relax_factor", 1e-8)
    set_optimizer_attribute(model, "mumps_mem_percent",  20)

    @variable(model, x[1:n, 1:2] >= 0)
    @variable(model, theta[i=1:n, j=i+1:n])
    @variable(model, R[i=1:n, j=i+1:n] >= 1e-4)

    # Warm-start
    for i in 1:n, d in 1:2
        set_start_value(x[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        dx = x_init[i, 1] - x_init[j, 1]
        dy = x_init[i, 2] - x_init[j, 2]
        set_start_value(theta[i, j], atan(dy, dx))
        set_start_value(R[i, j], max(norm([dx, dy]), 1e-4))
    end

    # Brisure de symétrie pour n=2 (système sous-déterminé sans ça)
    if n == 2
        fix(x[1, 1], x_init[1, 1]; force=true)
        fix(x[1, 2], x_init[1, 2]; force=true)
        fix(x[2, 2], x_init[2, 2]; force=true)
    end

    @variable(model, error_pair[i=1:n, j=i+1:n])

    for i in 1:n, j in i+1:n
        @NLconstraint(model, x[i, 1] - x[j, 1] == R[i, j] * cos(theta[i, j]))
        @NLconstraint(model, x[i, 2] - x[j, 2] == R[i, j] * sin(theta[i, j]))
        @NLconstraint(model, error_pair[i, j] == (c6 / R[i, j]^6) - Q[i, j])
    end

    @objective(model, Min, sum(error_pair[i, j]^2 for i in 1:n for j in i+1:n))

    optimize!(model)

    status = termination_status(model)
    statuts_valides = [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.SLOW_PROGRESS]

    if status in statuts_valides
        return (status, objective_value(model), value.(x), value.(R), value.(theta))
    else
        return nothing
    end
end

# ============================================================
#                       BOUCLE PRINCIPALE
# ============================================================
for n in n_values
    println("\n=========================================")
    println(" EXÉCUTION : n = $n  |  restarts max = $(nb_restarts(n))")
    println("=========================================")

    t0 = time()

    # Génération de la cible
    x_vrais = L * rand(n, limit_dim)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        d = max(norm(x_vrais[i, :] - x_vrais[j, :]), 1e-5)
        Q[i, j] = c6 / d^6
        Q[j, i] = Q[i, j]
    end

    # --- MULTI-START ---
    best_obj     = Inf
    best_status  = nothing
    best_x       = nothing
    best_R       = nothing
    best_theta   = nothing
    n_restarts_done = 0
    n_max = nb_restarts(n)

    for restart in 1:n_max
        x_init = generer_init(n, L, limit_dim)

        # On affiche Ipopt seulement au premier restart pour suivre la convergence
        res = resoudre_instance(n, Q, c6, L, limit_dim, x_init; verbose=(restart == 1))

        if res !== nothing
            stat, obj_val, x_sol, R_sol, theta_sol = res
            n_restarts_done += 1
            println("  Restart $restart/$n_max → obj = $(round(obj_val, sigdigits=6))  [$stat]")

            if obj_val < best_obj
                best_obj    = obj_val
                best_status = stat
                best_x      = x_sol
                best_R      = R_sol
                best_theta  = theta_sol
            end

            # Arrêt anticipé si on a trouvé une solution quasi-parfaite
            if best_obj < 1e-10
                println("  → Solution quasi-parfaite trouvée, arrêt anticipé.")
                break
            end
        else
            println("  Restart $restart/$n_max → échec ($( termination_status(
                let m = Model(Ipopt.Optimizer); m; end
            )))")
        end
    end

    t_exec = time() - t0

    # --- VALIDATION FINALE ---
    if best_x !== nothing
        val_rydberg = rydberg(best_x, Q, c6, n, limit_dim)
        val_polaire = verif_coherence_polaire(best_x, best_R, best_theta, n)
    else
        best_obj    = NaN
        best_status = "ÉCHEC_TOTAL"
        val_rydberg = NaN
        val_polaire = NaN
    end

    println(" → MEILLEUR obj          : $best_obj")
    println(" → Statut                : $best_status")
    println(" → Erreur Rydberg        : $val_rydberg")
    println(" → Erreur polaire max    : $val_polaire")
    println(" → Temps total           : $(round(t_exec, digits=3)) s")

    push!(results, (
        n,
        "Polaire_MS",          # MS = Multi-Start
        best_obj,
        string(best_status),
        val_rydberg,
        val_polaire,
        n_restarts_done,
        t_exec
    ))
end

println("\n=========================================")
println("          TABLEAU COMPARATIF FINAL       ")
println("=========================================")
println(results)