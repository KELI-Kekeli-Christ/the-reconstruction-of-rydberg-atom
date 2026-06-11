using JuMP
using Ipopt
using LinearAlgebra
using Random

# ====================================================================
# 1. FONCTIONS UTILITAIRES
# ====================================================================
function generate_unique_points_2d(n::Int, L::Float64)
    x = zeros(Float64, n, 2)
    min_dist = 1.0
    for i in 1:n
        valid = false
        while !valid
            candidate = L * rand(2)
            valid = true
            for j in 1:(i-1)
                if norm(candidate - x[j,:]) < min_dist
                    valid = false; break
                end
            end
            if valid; x[i,:] = candidate; end
        end
    end
    return x
end

# ====================================================================
# 2. ANALYSE THÉORIQUE (Diagonalisation)
# ====================================================================
function complete_diagonalization_analysis()
    println("\n=== Analyse de la Diagonalisation Orthogonale (Équations 11-14) ===")
    Q2 = zeros(Float64, 8, 8)
    # Remplissage pour contrainte 4
    Q2[5,5]=2; Q2[6,6]=2; Q2[5,6]=-2; Q2[6,5]=-2
    Q2[7,7]=2; Q2[8,8]=2; Q2[7,8]=-2; Q2[8,7]=-2
    
    b = zeros(8); b[4] = -1.0 
    decomp = eigen(Symmetric(Q2))
    D = Diagonal(decomp.values)
    V = decomp.vectors
    V_b = V * b 
    
    println("Valeurs propres (λ) identifiées :")
    display(round.(D.diag, digits=4))
end

# ====================================================================
# 3. RÉSOLUTION NUMÉRIQUE (Ipopt avec Pénalisation)
# ====================================================================
function solve_optimization_model(n::Int, q_matrix::Matrix{Float64}, C6::Float64)
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 1, "tol" => 1e-4))

    # Initialisation des variables avec start=1.0 pour éviter les divisions par zéro
    @variable(model, r[1:n, 1:n], start=1.0)
    @variable(model, w[1:n, 1:n], start=1.0)
    @variable(model, z[1:n, 1:n], start=1.0)
    @variable(model, mu[1:n, 1:n], start=1.0)
    @variable(model, x[1:n, 1:2], start=0.0)

    # Gabarits matriciels 8x8
    Q3 = zeros(8,8); Q3[1,2]=1.0; Q3[2,1]=1.0; b3 = zeros(8)
    Q6 = zeros(8,8); Q6[4,4]=2.0; b6 = zeros(8); b6[3]=-1.0
    Q5 = zeros(8,8); Q5[3,4]=1.0; Q5[4,3]=1.0; b5 = zeros(8); b5[2]=-1.0
    Q4 = zeros(8,8); Q4[5,5]=2; Q4[6,6]=2; Q4[5,6]=-2; Q4[6,5]=-2; Q4[7,7]=2; Q4[8,8]=2; Q4[7,8]=-2; Q4[8,7]=-2; b4 = zeros(8); b4[4]=-1.0

    # Fonction objectif avec pénalité P = 1000 pour forcer la réalisation des contraintes
    P = 1000.0
    obj_main = sum((r[i,j] - q_matrix[i,j])^2 for i in 1:(n-1) for j in (i+1):n)
    penalty = AffExpr(0.0)

    for i in 1:(n-1), j in (i+1):n
        X_ij = [r[i,j], w[i,j], mu[i,j], z[i,j], x[i,1], x[j,1], x[i,2], x[j,2]]
        
        # Pénalisation quadratique des contraintes
        penalty += (0.5 * (X_ij' * Q3 * X_ij) + dot(b3, X_ij) - C6)^2
        penalty += (0.5 * (X_ij' * Q6 * X_ij) + dot(b6, X_ij))^2
        penalty += (0.5 * (X_ij' * Q5 * X_ij) + dot(b5, X_ij))^2
        penalty += (0.5 * (X_ij' * Q4 * X_ij) + dot(b4, X_ij))^2
    end
    
    @objective(model, Min, obj_main + P * penalty)

    # Fixations spatiales
    fix(x[1,1], 0.0; force=true); fix(x[1,2], 0.0; force=true); fix(x[2,2], 0.0; force=true)
    
    optimize!(model)
    return model
end

# ====================================================================
# 4. EXÉCUTION DU FLUX
# ====================================================================
complete_diagonalization_analysis()

n = 4; C6 = 1; L = 100
x_ref = generate_unique_points_2d(n, L)
q_target = zeros(Float64, n, n)
for i in 1:(n-1), j in (i+1):n
    d = sqrt(max(sum((x_ref[i,:] - x_ref[j,:]).^2), 1e-6))
    val = C6 / (d^6)
    q_target[i,j] = val; q_target[j,i] = val
end

println("\n=== Lancement de l'optimisation (Matricielle + Pénalisée) ===")
model = solve_optimization_model(n, q_target, C6)

if termination_status(model) in [LOCALLY_SOLVED, OPTIMAL, ALMOST_LOCALLY_SOLVED, SLOW_PROGRESS, ITERATION_LIMIT]
    println("\nOptimisation réussie.")
    println("Valeur de l'objectif : ", round(objective_value(model), digits=6))
else
    println("Le solveur n'a pas convergé. Status: ", termination_status(model))
end