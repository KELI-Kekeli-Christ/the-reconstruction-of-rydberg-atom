using Ipopt, JuMP, LinearAlgebra, Random # Ajout de Random pour la reproductibilité si besoin

limit_dim = 2
limit_points = 20
number_generation_per_instance = 10
L = 10.0
C6 = 1

for inst in 1:number_generation_per_instance
    for n in 3:limit_points
        for d in 2:limit_dim
            println("\n--- Instance $inst : n = $n, dim = $d ---")
            
            # 1. GÉNÉRATION DES POSITIONS (Une seule fois par instance)
            x_true = L .* rand(n, d)
            
            # Calcul de la matrice cible Q
            Q = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    r2 = sum((x_true[i, k] - x_true[j, k])^2 for k in 1:d)
                    Q[i, j] = C6 / (r2^3)
                    Q[j, i] = Q[i, j]
                end
            end
            
            # 2. CONSTRUCTION DU MODÈLE
            model = Model(Ipopt.Optimizer)
            set_silent(model)

            @variable(model, x[1:n, 1:d] >= 0.0) # Bornes minimales pour éviter le négatif
            @variable(model, r[1:n, 1:n])       # Matrice complète pour simplifier JuMP

            # 3. INITIALISATION (Warm-start propre)
            x_init = L .* rand(n, d)
            for i in 1:n, k in 1:d
                set_start_value(x[i, k], x_init[i, k])
            end
            
            for i in 1:n, j in (i+1):n
                r2_init = sum((x_init[i, k] - x_init[j, k])^2 for k in 1:d)
                r_init_val = C6 / (r2_init^3)
                set_start_value(r[i, j], r_init_val)
            end

            # 4. FONCTION OBJECTIF
            @objective(model, Min, sum((r[i, j] - Q[i, j])^2 for i in 1:n, j in (i+1):n))

            # 5. CONTRAINTE DE DISTANCE NON LINÉAIRE (Flexible selon la dimension 'd')
            # Formulation stable : r[i,j] * (distance^2)^3 == C6
            for i in 1:n
                for j in (i+1):n
                    @constraint(model, r[i, j] * (sum((x[i, k] - x[j, k])^2 for k in 1:d)^3) == C6)
                end
            end
            
            # OPTIMISATION
            optimize!(model)
            
            # Récupération sécurisée du statut et de l'objectif
            status = termination_status(model)
            obj = objective_value(model)
            println("obj = ", round(obj, digits=7), " | status = ", status)
            end
        end
    end
end