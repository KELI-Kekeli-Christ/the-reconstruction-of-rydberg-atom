using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

L = 10.0
limit_dim = 2
limit_points = 100
results = []

for n in 3:limit_points
    for dim in 2:limit_dim

        # Générer une instance vraie
        x_true = L .* rand(n, dim)

        # Stocker Q comme distances au CARRÉ (définition EDGP)
        Q = zeros(n, n)
        for i in 1:n, j in i+1:n
            d2 = sum((x_true[i, k] - x_true[j, k])^2 for k in 1:dim)
            Q[i, j] = d2
            Q[j, i] = d2
        end

        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, x[1:n, 1:dim])
        @variable(model, y[i=1:n, j=i+1:n])
        @variable(model, z[i=1:n, j=i+1:n])

        # Initialisation cohérente
        x_init = L .* rand(n, dim)
        for i in 1:n, k in 1:dim
            set_start_value(x[i, k], x_init[i, k])
        end
        for i in 1:n, j in i+1:n
            set_start_value(y[i, j], (x_init[i,1] - x_init[j,1])^2)
            set_start_value(z[i, j], (x_init[i,2] - x_init[j,2])^2)
        end

        # Objectif : norme de Q(x) - Q
        @objective(model, Min,
            sum((y[i,j] + z[i,j] - Q[i,j])^2 for i in 1:n, j in i+1:n))

        # Contraintes d'égalité non-linéaires
        @constraint(model, [i in 1:n, j in i+1:n],
            y[i, j] == (x[i, 1] - x[j, 1])^2)
        @constraint(model, [i in 1:n, j in i+1:n],
            z[i, j] == (x[i, 2] - x[j, 2])^2)

        optimize!(model)

        obj = objective_value(model)
        status = termination_status(model)
        println("n=$n | dim=$dim | status=$status | obj=$obj")
        push!(results, (n=n, dim=dim, obj=obj, status=string(status)))
    end
end

# Sauvegarder résultats
df = DataFrame(results)
#CSV.write("resultats_edgp.csv", df)