using JuMP
using Ipopt
using LinearAlgebra
using Random
using Printf

# Constantes révisées pour correspondre à la réalité physique d'un QPU (en micromètres)
const C6 = 8600.0   # Valeur typique pour un état de Rydberg de Pasqal
const L  = 50.0     # Taille de la zone de piégeage
const R_min = 4.0   # Distance minimale de sécurité physique entre deux atomes

limit_points = 100
limit_dim = 2
number_generation_per_instance = 1 

Random.seed!(42) 

for n in 3:limit_points
    for dim in 2:limit_dim
        for _ in 1:number_generation_per_instance
            println("\n===  INSTANCE : n = $n, dim = $dim ===")
                
            # --- CORRECTION MULTI-ATOME : Génération saine sans collision initiale ---
            x_reel = zeros(n, dim)
            for i in 1:n
                while true
                    proposition = L * rand(dim)
                    # On s'assure que le nouvel atome est à au moins R_min de tous les précédents
                    if i == 1 || all(norm(proposition - x_reel[j, :]) >= R_min for j in 1:(i-1))
                        x_reel[i, :] = proposition
                        break
                    end
                end
            end

            # Calcul de la matrice cible Q issue de ces positions valides
            Q = zeros(n, n)
            for i in 1:n
                for j in (i+1):n
                    norm_value = norm(x_reel[i, :] - x_reel[j, :])
                    Q[i, j] = C6 / (norm_value^6)
                    Q[j, i] = Q[i, j]
                end
            end

            model = Model(Ipopt.Optimizer)
            set_optimizer_attribute(model, "print_level", 0)
            set_optimizer_attribute(model, "tol", 1e-6)

            @variable(model, x_opt[1:n, 1:dim] >= 0.0)
            
            # 1. CONTRAINTE PHYSIQUE : Distance minimale entre chaque atome
            for i in 1:n
                for j in (i+1):n
                    # Formule équivalente à : norme^2 >= R_min^2 (évite la racine carrée instable)
                    @constraint(model, sum((x_opt[i, k] - x_opt[j, k])^2 for k in 1:dim) >= R_min^2)
                end
            end
            
            # 2. FONCTION OBJECTIF : Moindres carrés directs (plus stable que le log)
            @objective(model, Min, sum(
                (
                    C6 / (sum((x_opt[i, k] - x_opt[j, k])^2 for k in 1:dim)^3) 
                    - Q[i, j]
                )^2 
                for i in 1:n, j in (i+1):n
            ))
            
            # Initialisation proche de la solution réelle pour guider efficacement le gradient
            for i in 1:n, k in 1:dim
                set_start_value(x_opt[i, k], x_reel[i, k] + randn() * 0.05) # Bruit légèrement réduit pour le warm-start
            end

            optimize!(model)
            
            err_finale = objective_value(model)
            println(" Valeur de l'erreur quadratique : ", round(err_finale, digits=7))
            
            # Seuil de tolérance adapté à l'erreur quadratique directe
            if err_finale < 1e-3
                println("  CONVERGENCE PARFAITE : Atomes placés sans collision (R >= $R_min) !")
                
                X_final = value.(x_opt)
                println("   Coordonnées finales des atomes X (Matrice $n × $dim) :")
                println("     " * "-"^30)
                for i in 1:n
                    @printf("     Atome %2d : [ ", i)
                    for d in 1:dim
                        @printf("%8.4f ", X_final[i, d])
                    end
                    println("]")
                end
                println("     " * "-"^30)
            else
                println("  Minimum local ou contrainte R_min stricte limitant l'ajustement parfait.")
            end
        end
    end
end