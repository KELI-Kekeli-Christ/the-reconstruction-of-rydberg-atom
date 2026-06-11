using JuMP
using Ipopt
using LinearAlgebra
using Random
using Statistics

# 1. Génération de la configuration initiale (sans collisions)
function generer_init(n, dim, L; seuil=1)
    x = zeros(n, dim)
    for i in 1:n
        x[i, :] = L * rand(dim)
        j = 1
        while j < i
            if norm(x[i, :] - x[j, :]) < seuil
                x[i, :] = L * rand(dim)
                j = 1
            else
                j += 1
            end
        end
    end
    return x
end

# 2. Calcul de l'erreur physique réelle (sur la vraie fonction r^6)
function erreur_rydberg_vraie(x_sol, Q_vrai, c6, n, dim)
    try
        val = 0.0
        for i in 1:n, j in i+1:n
            d = norm(x_sol[i,:] - x_sol[j,:])
            val += (c6 / d^6 - Q_vrai[i,j])^2
        end
        return val
    catch
        return Inf
    end
end

# --- FORMULATION 1 : L'ancienne méthode (Directe sur r^6) ---
function formulation_originale_r6(Q_bruite, c6, n, dim, x_init, L)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "tol", 1e-6) # Tolérance plus basse car r^6 est instable

    @variable(model, 0 <= x[1:n, 1:dim] <= L)
    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end

    # Objectif avec division par d^6 (comporte une singularité si d -> 0)
    @objective(model, Min, sum(
        ( c6 / (sum((x[i,d] - x[j,d])^2 for d in 1:dim)^3) - Q_bruite[i,j] )^2
        for i in 1:n for j in i+1:n
    ))
    @constraint(model, [i=1:n, j=i+1:n], sum((x[i,d] - x[j,d])^2 for d in 1:dim) >= 5)

    optimize!(model)
    return value.(x), termination_status(model)
end

# --- FORMULATION 2 : Votre méthode (Reformulée sur r^2) ---
function formulation_reformulee_r2(Q_bruite, c6, n, dim, x_init, L)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "tol", 1e-12)

    @variable(model, 0 <= x[1:n, 1:dim] <= L)
    for i in 1:n, d in 1:dim
        set_start_value(x[i,d], x_init[i,d])
    end

    # Objectif polynomial linéaire sans dénominateur
    @objective(model, Min, sum(
        ( sum((x[i,d] - x[j,d])^2 for d in 1:dim) - (c6 / Q_bruite[i,j])^(1/3) )^2
        for i in 1:n for j in i+1:n
    ))
    @constraint(model, [i=1:n, j=i+1:n], sum((x[i,d] - x[j,d])^2 for d in 1:dim) >= 5)

    optimize!(model)
    return value.(x), termination_status(model)
end







function rmse_distances(x_sol, x_vrai, n, dim)
    err = 0.0
    count = 0

    for i in 1:n, j in i+1:n
        d_sol = norm(x_sol[i,:] - x_sol[j,:])
        d_vrai = norm(x_vrai[i,:] - x_vrai[j,:])
        err += (d_sol - d_vrai)^2
        count += 1
    end

    return sqrt(err / count)
end


# --- Script de test de résistance ---
Random.seed!(1234) # Fixer le seed pour comparer équitablement
n = 20          # Légèrement réduit à 40 pour que le modèle r^6 ne crashe pas instantanément
c6 = 1
r=c6^(1/6)
L = r*sqrt(n)
dim = 2 



# A. Génération de la "Vraie" physique cachée
x_reel = generer_init(n, dim, L; seuil=1.1)
Q_pur = zeros(n, n)
for i in 1:n, j in i+1:n
    d = norm(x_reel[i,:] - x_reel[j,:])
    Q_pur[i,j] = c6 / d^6
    Q_pur[j,i] = Q_pur[i,j]
end

# B. Injection d'un bruit de 5% sur les données mesurées Q
niveau_bruit = 0
Q_bruite = copy(Q_pur)
for i in 1:n, j in i+1:n
    Q_bruite[i,j] *= max(1.0 + niveau_bruit * randn())
    Q_bruite[j,i] = Q_bruite[i,j]
end

# C. Point de départ identique (volontairement éloigné de la solution)
x_init = generer_init(n, dim, L; seuil=1.0)

println("=== BENCHMARK DE RÉSISTANCE AU BRUIT (Bruit: $(niveau_bruit*100)%) ===")
println("L optimal calculé : ", L)
# Calcul Modèle 1 (r^6)
x_sol_r6, status_r6 = formulation_originale_r6(Q_bruite, c6, n, dim, x_init, L)
err_r6 = erreur_rydberg_vraie(x_sol_r6, Q_pur, c6, n, dim)
println("\n[Formulation 1 : Ancienne (r⁶)]")
println("  Statut du solver     : ", status_r6)
println("  Erreur vs Réalité    : ", err_r6)
println("x_sol_r6: ", x_sol_r6)
# Calcul Modèle 2 (r^2)
x_sol_r2, status_r2 = formulation_reformulee_r2(Q_bruite, c6, n, dim, x_init, L)
err_r2 = erreur_rydberg_vraie(x_sol_r2, Q_pur, c6, n, dim)
println("\n[Formulation 2 : Nouvelle (r²)]")
println("  Statut du solver     : ", status_r2)
println("  Erreur vs Réalité    : ", err_r2)
println("x_sol_r2: ", x_sol_r2)

rmse_r6 = rmse_distances(x_sol_r6, x_reel, n, dim)
rmse_r2 = rmse_distances(x_sol_r2, x_reel, n, dim)

println("\n--- RMSE GEOMETRIQUE ---")
println("RMSE r6 : ", rmse_r6)
println("RMSE r2 : ", rmse_r2)