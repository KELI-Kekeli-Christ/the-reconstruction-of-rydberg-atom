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
            println("\n=========================================")
            println("ÉVALUATION MATRICIELLE : n = ", n, ", dim = ", dim)
            println("=========================================")
                
            # Nombre de paires uniques
            P = Int(n * (n - 1) / 2)
            
            # Génération des positions cibles initiales
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
            set_optimizer_attribute(model, "print_level", 0)

            # Déclaration de nos variables sous forme vectorielle pour coller à X
            @variable(model, r[i=1:n, j=(i+1):n])
            @variable(model, w[i=1:n, j=(i+1):n])
            @variable(model, u[i=1:n, j=(i+1):n])
            @variable(model, z[i=1:n, j=(i+1):n])
            @variable(model, x[i=1:n, d=1:dim])

            # Initialisation des coordonnées
            for i in 1:n
                for d in 1:dim
                    set_start_value(x[i, d], x_init[i, d])
                end
            end

            # Initialisation des variables de paires (r, w, u, z) en cascade physique
            for i in 1:n
                for j in (i+1):n
                    d_sq = sum((x_init[i, k] - x_init[j, k])^2 for k in 1:dim)
                    w_val = d_sq
                    u_val = w_val^3
                    r_val = C6 / (u_val + 1e-6)
                    z_val = u_val
                    set_start_value(w[i, j], w_val)
                    set_start_value(u[i, j], u_val)
                    set_start_value(z[i, j], z_val)
                    set_start_value(r[i, j], r_val)
                end
            end


            set_optimizer_attribute(model, "tol", 1e-12)                     # Demande une précision ultra-fine (par défaut 1e-8)
            set_optimizer_attribute(model, "dual_inf_tol", 1e-10)            # Force le respect strict des critères d'optimalité
            set_optimizer_attribute(model, "constr_viol_tol", 1e-10)        # Force le respect strict de ta grande contrainte quadrique
            set_optimizer_attribute(model, "acceptable_tol", 1e-11)





            # Définition de l'objectif (Moindres carrés sur les résidus)
            @objective(model, Min, sum((r[i, j] - Q_cible[i, j])^2 for i in 1:n, j in (i+1):n))

            # =================================================================
            # CONtruction DE NOTRE STRUCTURE MATRICIELLE FINALE f(X) = 0
            # =================================================================
            
            # 1. Création du grand vecteur d'état X sous forme de tableau JuMP
            # On aplatit chaque paquet de variables de taille P ou n
            v_r = [r[i, j] for i in 1:n for j in (i+1):n]
            v_w = [w[i, j] for i in 1:n for j in (i+1):n]
            v_u = [u[i, j] for i in 1:n for j in (i+1):n]
            v_z = [z[i, j] for i in 1:n for j in (i+1):n]
            v_x1 = [x[i, 1] for i in 1:n]
            v_x2 = [x[i, 2] for i in 1:n]

            # On rassemble tout dans le vecteur colonne X global (Taille : 4P + 2n)
            X_global = [v_r; v_w; v_u; v_z; v_x1; v_x2]
            taille_X = length(X_global)

            # 2. Allocation de notre grande matrice Q et du vecteur b
            Q_mat = zeros(taille_X, taille_X)
            b_vec = zeros(taille_X)

            # Remplissage par bloc selon notre schéma pas à pas
            for k in 1:P
                # Blocs Contrainte 1 : r_ij * w_ij (Intersections r et w)
                Q_mat[k, P + k] = 0.5
                Q_mat[P + k, k] = 0.5

                # Blocs Contrainte 3 : u_ij * z_ij (Intersections u et z)
                Q_mat[2P + k, 3P + k] = 0.5
                Q_mat[3P + k, 2P + k] = 0.5
                b_vec[P + k] = -1.0  # Terme linéaire -w_ij

                # Blocs Contrainte 4 : z_ij^2
                Q_mat[3P + k, 3P + k] = 1.0  # Carré pur sur la diagonale de z
                b_vec[2P + k] = -1.0  # Terme linéaire -u_ij
                
                # Terme linéaire -z_ij issu de la contrainte 2
                b_vec[3P + k] = -1.0
            end

            # Blocs Contrainte 2 : Laplacien des coordonnées (x_i - x_j)^2
            # On reconstruit le motif (1, -1 / -1, 1) pour chaque paire sur l'axe 1 et l'axe 2
            paire_idx = 1
            for i in 1:n
                for j in (i+1):n
                    # Axe 1 (x[:,1])
                    idx_xi1 = 4P + i
                    idx_xj1 = 4P + j
                    Q_mat[idx_xi1, idx_xi1] += 1.0
                    Q_mat[idx_xj1, idx_xj1] += 1.0
                    Q_mat[idx_xi1, idx_xj1] -= 1.0
                    Q_mat[idx_xj1, idx_xi1] -= 1.0

                    # Axe 2 (x[:,2])
                    idx_xi2 = 4P + n + i
                    idx_xj2 = 4P + n + j
                    Q_mat[idx_xi2, idx_xi2] += 1.0
                    Q_mat[idx_xj2, idx_xj2] += 1.0
                    Q_mat[idx_xi2, idx_xj2] -= 1.0
                    Q_mat[idx_xj2, idx_xi2] -= 1.0
                    
                    paire_idx += 1
                end
            end

            # Constante globale c = -P * C6
            c = -P * C6

            # =================================================================
            # INJECTION DE L'ÉQUATION QUADRIQUE GLOBALE UNIQUE DANS JuMP
            # =================================================================
            
            # f(X) = X^T * Q * X + b^T * X + c == 0
            @constraint(model, X_global' * Q_mat * X_global + b_vec' * X_global + c == 0)

            # Résolution numérique par Ipopt
            optimize!(model)
       
            # Affichage des résultats
            println("Statut de convergence : ", termination_status(model))
            println("Valeur de l'objectif  : ", objective_value(model))
        end
    end
end