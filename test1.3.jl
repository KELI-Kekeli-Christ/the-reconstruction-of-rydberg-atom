using LinearAlgebra, Statistics, Random, Printf, Plots

function solve_mds(n, dim, Q, c6)
    # Étape 1 : distances cibles
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        d_ij = (c6 / Q[i,j])^(1/6)
        D[i,j] = d_ij
        D[j,i] = d_ij
    end

    # Étape 2 : double centrage → matrice de Gram
    D2 = D .^ 2
    H  = I - ones(n,n) / n
    B  = Symmetric(-0.5 * H * D2 * H)

    # Étape 3 : décomposition spectrale
    F     = eigen(B)
    λ     = F.values
    V     = F.vectors

    idx   = sortperm(λ, rev=true)[1:dim]
    λ_top = max.(λ[idx], 0.0)

    # Étape 4 : positions
    X_mds = V[:, idx] * Diagonal(sqrt.(λ_top))

    # centrage
    X_mds .-= mean(X_mds, dims=1)

    return D, B,λ_top, λ, X_mds
end


# =========================
# CHECK RECONSTRUCTION
# =========================
function check_reconstruction(D, X)
    n = size(D,1)
    err_max = 0.0

    for i in 1:n, j in i+1:n
        d_hat = norm(X[i,:] - X[j,:])
        err_max = max(err_max, abs(D[i,j] - d_hat))
    end

    return err_max
end


# =========================
# CHECK X (NOUVEAU)
# =========================
function check_X_proximity(X, tol=1e-8)
    n = size(X,1)
    min_dist = Inf
    too_close = false

    for i in 1:n, j in i+1:n
        d = norm(X[i,:] - X[j,:])
        if d < min_dist
            min_dist = d
        end
        if d < tol
            too_close = true
        end
    end

    return min_dist, too_close
end


# =========================
# NOISE
# =========================
function add_noise(Q, sigma)
    Qn = copy(Q)
    for i in 1:size(Q,1), j in i+1:size(Q,1)
        noise = 1.0 + sigma * randn()
        Qn[i,j] *= noise
        Qn[j,i] = Qn[i,j]
    end
    return Qn
end


# =========================
# SCRIPT TEST
# =========================
n = 3000
c6 = 1.0
L = 1
dim = 2

x_ref = rand(n, dim) * L

for i in 1:n, j in i+1:n
    while norm(x_ref[i,:] - x_ref[j,:]) < 1e-8
        x_ref[i,:] = rand(dim) * L
    end
end

Q = zeros(Float64, n, n)
for i in 1:n, j in i+1:n
    d = norm(x_ref[i,:] - x_ref[j,:])
    Q[i,j] = c6 / (d^6)
    Q[j,i] = Q[i,j]
end


# =========================
# TESTS BRUIT + AFFICHAGE
# =========================
for sigma in [0.0, 0.01, 0.05, 0.1]

    println("\n====================================")
    println("BRUIT σ = ", sigma)

    Qn = add_noise(Q, sigma)

    D, B, λ_top,λ, X_mds = solve_mds(n, dim, Qn, c6)

    err = check_reconstruction(D, X_mds)
    min_dist, too_close = check_X_proximity(X_mds, 1e-8)

    println("\n--- Résultats MDS ---")

    #println("Distances D :")
    #display(round.(D, digits=4))

    #println("\nMatrice de Gram B :")
    #display(round.(B, digits=4))

    println("\nValeurs propres λ :")
    #display(round.(λ, digits=4))

    #println("\nPositions X_mds :")
    #display(round.(X_mds, digits=4))

    println("\n--- Checks ---")
    println("Max erreur reconstruction D vs X : ", err)
    println("Distance minimale entre points : ", min_dist)

    if too_close
        println("⚠ WARNING : points trop proches (< 1e-8)")
    else
        println("✔ OK : pas de points trop proches")
    end

    println("valeurs propres : ", sort(λ, rev=true)[1:3])
    println("Valeurs λ_top utilisées : ", λ_top)




# on prend un sous-échantillon si n est grand
idx = n > 2000 ? rand(1:n, 100) : 1:n

p = scatter(x_ref[idx,1], x_ref[idx,2], label="Original", alpha=0.5)
    scatter!(p, X_mds[idx,1], X_mds[idx,2], label="MDS", alpha=0.5)
    title!(p, "Original vs MDS (sans alignement), σ = $sigma")
    
    # Commande critique pour l'affichage dans une boucle
    display(p)
    savefig(p, "mds_comparison_sigma_$(Int(round(sigma*100))).png")

title!("Original vs MDS (sans alignement), σ = $sigma")
end