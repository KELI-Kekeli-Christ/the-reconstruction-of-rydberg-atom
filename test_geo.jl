using JuMP
using Ipopt
using LinearAlgebra
using DataFrames

# --- CONFIGURATION DES PARAMÈTRES ---
n_values  = [2, 3]  # Nombre de points à tester
limit_dim = 2           # Dimensionnalité fixe (2D pour le cercle)
L         = 10.0       # Taille du domaine de recherche
c6        = 1.0         # Constante de ta contrainte

# DataFrame pour stocker la synthèse des performances
results = DataFrame(
    n           = Int[],
    formulation = String[],
    obj         = Float64[],
    status      = String[],
    time_s      = Float64[]
)

for n in n_values
    println("\n=========================================")
    println(" EXÉCUTION DU MODÈLE : n = $n points")
    println("=========================================")

    # --- 1. GÉNÉRATION DES POINTS INITIAUX GÉOMÉTRIQUES ---
    x_init = zeros(n, limit_dim)
    min_dist = 0.5
    for i in 1:n
        accepted = false
        while !accepted
            candidate = L * rand(limit_dim)
            accepted = true
            for j in 1:(i-1)
                if norm(candidate - x_init[j, :]) < min_dist
                    accepted = false
                    break
                end
            end
            if accepted
                x_init[i, :] = candidate
            end
        end
    end

    # --- 2. CALCUL DE LA MATRICE CIBLE Q_ij (Données du problème) ---
    x_reel=rand(n, limit_dim) * L  # Configuration cible pour calculer Q (différente de x_init)
    Q = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            d = norm(x_reel[i, :] - x_reel[j, :])
            Q[i, j] = c6 / (d^6)
            Q[j, i] = Q[i, j]
        end
    end

    # --- 3. INITIALISATION DU MODÈLE JUMP AVEC IPOPT ---
    t0 = time()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0) # Mode silencieux
    set_optimizer_attribute(model, "tol", 1e-12)     # Haute précision

    # --- 4. VARIABLES DE DÉCISION ---
    @variable(model, x[1:n, 1:2] >= 0)
    
    # Variables angulaires libres θ_ij pour chaque paire (Section 6 de ta démo)
    @variable(model, theta[1:n, 1:n])

    for i in 1:n, d in 1:2
        set_start_value(x[i, d], x_init[i, d])
    end
    for i in 1:n, j in i+1:n
        if i != j
            # Calcul de l'angle initial à partir des positions générées
            dx = x_init[i, 1] - x_init[j, 1]
            dy = x_init[i, 2] - x_init[j, 2]
            set_start_value(theta[i, j], atan(dy, dx))
        end
    end

    # Somme des moindres carrés de l'objectif : ∑ [ c6/(VᵀDV) - Q_ij ]²
    # Pour l'écrire proprement, on crée une variable d'écart objectif par paire
    @variable(model, error_pair[i=1:n, j=i+1:n])

    for i in 1:n
        for j in i+1:n
            # Définition des écarts élémentaires (Section 3)
            z1 = x[i, 1] - x[j, 1]
            z2 = x[i, 2] - x[j, 2]

            # Base réduite non-redondante V (Section 4)
            # V = [z1³, z1²z2, z1z2², z2³]ᵀ
            # Matrice D diagonale intégrée directement dans le produit VᵀDV :
            V_D_V = (z1^3)^2 + 3 * (z1^2 * z2)^2 + 3 * (z1 * z2^2)^2 + (z2^3)^2

            # R = (VᵀDV)^(1/6)
            R = (V_D_V)^(1/6)

            # Équation de résolution du cercle découpé (Système 7 de ta démo)
            @constraint(model, z1 == R * cos(theta[i, j]))
            @constraint(model, z2 == R * sin(theta[i, j]))

            # Liaison avec la fonction objectif via la variable d'erreur r_ij
            # r_ij = c6 / VᵀDV
            @constraint(model, error_pair[i, j] == (c6 / V_D_V) - Q[i, j])
        end
    end

    # --- 7. FONCTION OBJECTIF ---
    @objective(model, Min, sum(error_pair[i, j]^2 for i in 1:n for j in i+1:n))

    # --- 8. RÉSOLUTION ---
    optimize!(model)

    # --- 9. ENREGISTREMENT ET AFFICHAGE DES RÉSULTATS ---
    t_exec = time() - t0
    obj_val = objective_value(model)
    status = termination_status(model)

    println(" -> Statut de convergence : $status")
    println(" -> Valeur objective      : $obj_val")
    println(" -> Temps d'exécution     : $(round(t_exec, digits=4)) secondes")

    push!(results, (n, "Polaire_Cercle_VᵀDV", obj_val, string(status), t_exec))
end

println("\n=========================================")
println("          TABLEAU COMPARATIF FINALE      ")
println("=========================================")
println(results)

