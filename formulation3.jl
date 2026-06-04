using JuMP
using Ipopt
using Random
using LinearAlgebra

# Paramètres globaux
limit_points = 10
limit_dim = 2
number_generation_per_instance = 1
L = 10
C6 = 1

for n in 3:limit_points
    for dim in 2:limit_dim
        for _ in 1:number_generation_per_instance
            println("\n=========================================")
            println("ÉVALUATION : n = ", n, ", dim = ", dim)
            println("=========================================")
                
            #Génération des positions de départ (Renommé pour éviter le conflit avec JuMP)
            x_init = L * rand(n, dim)

            # Calcul de la matrice cible Q_cible
            Q_cible = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    norm_value = norm(x_init[i, :] - x_init[j, :])
                    Q_cible[i, j] = C6 / (norm_value)^6
                    Q_cible[j, i] = Q_cible[i, j]
                end
            end

            # Initialisation du modèle JuMP
            model = Model(Ipopt.Optimizer)
            set_optimizer_attribute(model, "print_level", 0) # Mode silencieux

            # Déclaration des variables restreintes à la partie supérieure stricte (i < j)
            @variable(model, x[i=1:n, d=1:dim])
            @variable(model, r[i=1:n, j=(i+1):n])
            @variable(model, w[i=1:n, j=(i+1):n])
            @variable(model, z[i=1:n, j=(i+1):n])
            @variable(model, u[i=1:n, j=(i+1):n])

            # Initialisation (Warm-start) pour aider Ipopt à converger rapidement
            for i in 1:n
                for d in 1:dim
                    set_start_value(x[i, d], x_init[i, d])
                end
            end

            # Fonction objectif : Moindres carrés sur le résidu r par rapport à Q_cible
            @objective(model, Min, sum((r[i, j] - Q_cible[i, j])^2 for i in 1:n, j in (i+1):n))

            # --- APPLICATION DE NOTRE SYSTEME DE MATRICES QUADRATIQUES ---

            # Contrainte 1 : r_ij * w_ij == C6
            @constraint(model, [i=1:n, j=(i+1):n], r[i, j] * w[i, j] == C6)
            
            # Contrainte 2 : Norme euclidienne au carré (Généralisée pour s'adapter à dim)
            @constraint(model, [i=1:n, j=(i+1):n], z[i, j] == sum((x[i, d] - x[j, d])^2 for d in 1:dim))
                    
            # Contrainte 3 : w_ij == u_ij * z_ij
            @constraint(model, [i=1:n, j=(i+1):n], w[i, j] == u[i, j] * z[i, j])
            
            # Contrainte 4 : u_ij == z_ij^2
            @constraint(model, [i=1:n, j=(i+1):n], u[i, j] == z[i, j]^2)

            # Résolution numérique du système
            optimize!(model)
       
            # Affichage condensé des résultats de l'instance
            println("Statut de convergence : ", termination_status(model))
            println("Valeur de l'objectif  : ", objective_value(model))
            
            if termination_status(model) == MOI.OPTIMAL
                println("Résultat : Le solveur a trouvé un agencement parfait.")
            else
                println("Résultat : Échec ou approximation locale.")
            end
        end
    end
end