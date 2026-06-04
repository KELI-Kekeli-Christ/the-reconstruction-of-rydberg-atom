using JuMP
using Ipopt
using Random
using LinearAlgebra

# Paramètres globaux 
limit_points = 100
limit_dim = 2
number_generation_per_instance = 1
L = 10
C6 = 1

for n in 100:limit_points
    for dim in 2:limit_dim
        for _ in 1:number_generation_per_instance
            println("\n==================================================")
            println("ÉVALUATION PAR DIAGONALISATION : n = ", n, ", dim = ", dim)
            println("==================================================")
                
            P = Int(n * (n - 1) / 2)
            taille_X = 4 * P + 2 * n
            
            # 1. Génération des positions cibles initiales
            x_init = L * rand(n, dim)
            Q_cible = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    norm_value = norm(x_init[i, :] - x_init[j, :])
                    Q_cible[i, j] = C6 / (norm_value)^6
                    Q_cible[j, i] = Q_cible[i, j]
                end
            end

            # 2. Construction de la structure quadrique d'origine (Q, b, c)
            Q_originale = zeros(taille_X, taille_X)
            b_original = zeros(taille_X)

            for k in 1:P
                Q_originale[k, P + k] = 0.5
                Q_originale[P + k, k] = 0.5
                Q_originale[2P + k, 3P + k] = 0.5
                Q_originale[3P + k, 2P + k] = 0.5
                b_original[P + k] = -1.0  # -w_ij
                Q_originale[3P + k, 3P + k] = 1.0  # z_ij^2
                b_original[2P + k] = -1.0  # -u_ij
                b_original[3P + k] = -1.0  # -z_ij
            end

            for i in 1:n
                for j in (i+1):n
                    idx_xi1, idx_xj1 = 4P + i, 4P + j
                    Q_originale[idx_xi1, idx_xi1] += 1.0
                    Q_originale[idx_xj1, idx_xj1] += 1.0
                    Q_originale[idx_xi1, idx_xj1] -= 1.0
                    Q_originale[idx_xj1, idx_xi1] -= 1.0

                    idx_xi2, idx_xj2 = 4P + n + i, 4P + n + j
                    Q_originale[idx_xi2, idx_xi2] += 1.0
                    Q_originale[idx_xj2, idx_xj2] += 1.0
                    Q_originale[idx_xi2, idx_xj2] -= 1.0
                    Q_originale[idx_xj2, idx_xi2] -= 1.0
                end
            end
            c = -P * C6

            # =================================================================
            # 3. INTERSECT THÉORIQUE : DIAGONALISATION ORTHOGONALE (Q = V^T D V)
            # =================================================================
            println("Calcul de la décomposition spectrale (Eigen)...")
            
            # Extraction des valeurs propres (D) et de la matrice de passage (V)
            # Pour respecter Q = V^T * D * V avec V orthogonale :
            decomp = eigen(Symmetric(Q_originale)) 
            valeurs_propres = decomp.values       # Contient la diagonale de D
            V = decomp.vectors'                   # Matrice orthogonale V

            # Nouveau vecteur linéaire découplé : V_b = V * b
            V_b = V * b_original

            # =================================================================
            # 4. MODÉLISATION DANS L'ESPACE DÉCOUPLÉ (Y)
            # =================================================================
            model = Model(Ipopt.Optimizer)
            set_optimizer_attribute(model, "print_level", 0)

            # Déclaration de la variable découplée Y (Pas de bornes car espace virtuel)
            @variable(model, Y[1:taille_X])

            # Expression de la contrainte sous sa forme diagonale (Équation 14 de ton rapport)
            # 1/2 * Y^T * D * Y + (Vb)^T * Y + c == 0
            @constraint(model, 0.5 * sum(valeurs_propres[i] * Y[i]^2 for i in 1:taille_X) + V_b' * Y + c == 0)

            # Pour l'objectif, comme on cherche à coller à Q_cible, on doit reconstruire 
            # virtuellement le vecteur X d'origine à partir de Y pour extraire "r"
            # X = V^T * Y
            X_virtuel = V' * Y
            
            # Les P premières composantes de X_virtuel correspondent aux variables r_ij
            @objective(model, Min, sum((X_virtuel[k] - Q_cible[i, j])^2 
                for (k, (i, j)) in enumerate([(i, j) for i in 1:n for j in (i+1):n])))

            # =================================================================
            # 5. RÉSOLUTION ET RECONSTRUCTION DES VARIABLES PHYSIQUES
            # =================================================================
            optimize!(model)
       
            println("Statut de convergence : ", termination_status(model))
            
                # Extraction du Y optimal
                Y_opt = value.(Y)
                
                # Reconstruction du vrai vecteur physique X = V^T * Y
                X_physique = V' * Y_opt
                
                println("Valeur de l'objectif  : ", objective_value(model))
                println("Reconstruction réussie. Nombre de variables d'état : ", length(X_physique))
                println("Nombre de valeurs propres nulles : ", sum(iszero.(valeurs_propres)))
                println("Nombre de valeurs propres positives : ", sum(valeurs_propres .> 0))
                println("Nombre de valeurs propres négatives : ", sum(valeurs_propres .< 0))
                # Les variables physiques r, w, u, z, x sont maintenant disponibles dans X_physique
        end
    end
end