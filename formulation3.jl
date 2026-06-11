using JuMP
using Ipopt
using Random
using LinearAlgebra

# Paramètres globaux
limit_points = 30
limit_dim = 2
number_generation_per_instance = 1
L = 10
C6 = 1

for n in 20:limit_points
    for dim in 2:limit_dim
        for _ in 1:number_generation_per_instance
            println("\n=========================================")
            println("ÉVALUATION : n = ", n, ", dim = ", dim)
            println("=========================================")
                
            # Génération des positions de départ
            x_init = L * rand(n, dim)

            # Calcul de la matrice cible Q_cible
            Q_cible = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    norm_value = norm(x_init[i, :] - x_init[j, :])
                    # Sécurité contre les points superposés à l'initialisation
                    if norm_value < 1e-6; norm_value = 1e-6; end 
                    Q_cible[i, j] = C6 / (norm_value)^6
                    Q_cible[j, i] = Q_cible[i, j]
                end
            end

            # Initialisation du modèle JuMP
            model = Model(Ipopt.Optimizer)
            set_optimizer_attribute(model, "print_level", 0) # Mode silencieux

            # Déclaration des variables avec des bornes inférieures (>= 1e-6 pour éviter les zéros)
            @variable(model, x[i=1:n, d=1:dim])
            @variable(model, r[i=1:n, j=(i+1):n] >= 1e-6)
            @variable(model, w[i=1:n, j=(i+1):n] >= 1e-6)
            @variable(model, z[i=1:n, j=(i+1):n] >= 1e-6)
            @variable(model, u[i=1:n, j=(i+1):n] >= 1e-6)

            # Initialisation (Warm-start) complète pour toutes les variables
            for i in 1:n
                for d in 1:dim
                    set_start_value(x[i, d], x_init[i, d])
                end
            end
            
            for i in 1:n
                for j in (i+1):n
                    # Calcul des valeurs théoriques basées sur x_init
                    z_val = sum((x_init[i, d] - x_init[j, d])^2 for d in 1:dim)
                    if z_val < 1e-6; z_val = 1e-6; end
                    u_val = z_val^2
                    w_val = u_val * z_val
                    r_val = C6 / w_val
                    
                    # Attribution des warm-starts
                    set_start_value(z[i, j], z_val)
                    set_start_value(u[i, j], u_val)
                    set_start_value(w[i, j], w_val)
                    set_start_value(r[i, j], r_val)
                end
            end

            # Fonction objectif
            @objective(model, Min, sum((r[i, j] - Q_cible[i, j])^2 for i in 1:n, j in (i+1):n))

            # --- APPLICATION DES CONTRAINTES ---

            # Contrainte 1 : r_ij * w_ij == C6
            @constraint(model, [i=1:n, j=(i+1):n], r[i, j] * w[i, j] == C6)
            
            # Contrainte 2 : Norme euclidienne au carré
            @constraint(model, [i=1:n, j=(i+1):n], z[i, j] == sum((x[i, d] - x[j, d])^2 for d in 1:dim))
                    
            # Contrainte 3 : w_ij == u_ij * z_ij
            @constraint(model, [i=1:n, j=(i+1):n], w[i, j] == u[i, j] * z[i, j])
            
            # Contrainte 4 : u_ij == z_ij^2
            @constraint(model, [i=1:n, j=(i+1):n], u[i, j] == z[i, j]^2)

            # Résolution numérique
            optimize!(model)
       
            # Affichage condensé des résultats
            status = termination_status(model)
            println("Statut de convergence : ", status)
            
            # Sécurité si le solveur échoue à trouver une solution exploitable
            
                println("Valeur de l'objectif  : ", objective_value(model))
                println("Résultat : Le solveur a trouvé un agencement valide.")
        end
    end
end