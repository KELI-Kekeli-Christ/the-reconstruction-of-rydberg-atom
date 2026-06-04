using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

const C6 = 862690.0
const L  = 10.0
const λ  = 1.0    # paramètre Moreau-Yosida
limit_points = 20
n_starts     = 5
results      = []

for n in 3:limit_points

    x_true = L .* rand(n, 2)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        r2 = sum((x_true[i,k] - x_true[j,k])^2 for k in 1:2)
        Q[i,j] = C6 / r2^3
        Q[j,i] = Q[i,j]
    end

    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        D[i,j] = (C6 / Q[i,j])^(1/3)
        D[j,i] = D[i,j]
    end

    best_obj = 1e30

    for _ in 1:n_starts

        x_k = L .* rand(n, 2)

        for iter in 1:50

            # ── DCA : min f2(x) + λ/2 ‖x - x^k‖² ────────────────
            model = Model(Ipopt.Optimizer)
            set_silent(model)

            @variable(model, x[1:n, 1:2])
            for i in 1:n, k in 1:2
                set_start_value(x[i,k], x_k[i,k])
            end

            # f2(x) = Σ(r² - d*)²
            # terme proximal = λ/2 Σ(x - x^k)²
            @objective(model, Min,
                sum(((x[i,1]-x[j,1])^2 +
                     (x[i,2]-x[j,2])^2 - D[i,j])^2
                    for i in 1:n, j in i+1:n)
                + (λ/2) * sum((x[i,k] - x_k[i,k])^2
                    for i in 1:n, k in 1:2))

            optimize!(model)
            x_new = value.(x)

            # convergence
            diff = maximum(abs.(x_new - x_k))
            x_k  = x_new
            diff < 1e-10 && break
        end

        obj = sum(((x_k[i,1]-x_k[j,1])^2 +
                   (x_k[i,2]-x_k[j,2])^2 - D[i,j])^2
                  for i in 1:n, j in i+1:n)
        best_obj = min(best_obj, obj)
    end

    println("n=$n | best_obj=$(round(best_obj, digits=8))")
    push!(results, (n=n, obj=best_obj))
end

CSV.write("resultats_dca_prox.csv", DataFrame(results))