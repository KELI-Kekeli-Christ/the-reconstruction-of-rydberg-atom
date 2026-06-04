using Ipopt, JuMP, LinearAlgebra, Random

limit_dim = 2
limit_points = 20
number_generation_per_instance = 2
L = 10.0
C6 = 1.0 # Mieux vaut utiliser un flottant explicitement

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
            
            # 2. CONSTRUCTION DU MODÈLE POP
            model = Model(Ipopt.Optimizer)
            set_silent(model)
            
            # Paramètres pour aider à la convergence sur la surface non-linéaire
            set_optimizer_attribute(model, "max_iter", 5000)
            set_optimizer_attribute(model, "tol", 1e-6)

            @variable(model, x[1:n, 1:d] >= 0.0) 
            
            # ASTUCE 1 : On ne crée la variable r que pour les paires uniques (j > i)
            # ASTUCE 2 : r doit être positif (force physique et distance)
            @variable(model, r[i=1:n, j=(i+1):n] >= 0.0)

            # 3. INITIALISATION (Warm-start)
            # Excellente idée ici : r_init satisfait parfaitement la contrainte au t=0 !
            x_init = L .* rand(n, d)
            for i in 1:n, k in 1:d
                set_start_value(x[i, k], x_init[i, k])
            end
            
            for i in 1:n, j in (i+1):n
                r2_init = sum((x_init[i, k] - x_init[j, k])^2 for k in 1:d)
                r_init_val = C6 / (r2_init^3)
                set_start_value(r[i, j], r_init_val)
            end

            # 4. FONCTION OBJECTIF (Somme de carrés)
            @objective(model, Min, sum((r[i, j] - Q[i, j])^2 for i in 1:n, j in (i+1):n))

            # 5. CONTRAINTE DE DISTANCE (Égalité Polynomiale de degré 6)
            for i in 1:n
                for j in (i+1):n
                    @constraint(model, r[i, j] * (sum((x[i, k] - x[j, k])^2 for k in 1:d)^3) == C6)
                end
            end
            
            # OPTIMISATION
            optimize!(model)
            
            # Récupération sécurisée du statut et de l'objectif
            status = termination_status(model)
            
            # Sécurité pour éviter que la fonction objective_value ne plante si aucune solution n'est trouvée
                obj = objective_value(model)
                println("  obj = ", round(obj, digits=7), " | status = ", status)
            end
        end
    end
end