using Pkg
Pkg.add("StatsBase")
Pkg.add("CSV")
Pkg.add("DataFrames")

using JuMP, Ipopt, LinearAlgebra, StatsBase, CSV, DataFrames

# --- Paramètres ---
n_values  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
limit_dim = 2
n_starts  = 10
L         = 10
c6        = 1

results = DataFrame(
    n        = Int[],
    dim      = Int[],
    g_type   = String[],
    g_value  = Int[],
    best_obj = Float64[]
)

for n in n_values
    for dim in 2:limit_dim
        println("\n=== n = $n, dim = $dim ===")

        x = L * rand(n, dim)

        Q = zeros(n, n)
        for i in 1:n
            for j in i+1:n
                d = norm(x[i,:] - x[j,:])
                Q[i,j] = c6 / d^6
                Q[j,i] = c6 / d^6
            end
        end

        g_cases = [
            ("2sqrt(n)", ceil(Int, 2 * sqrt(n))),
            ("sqrt(n)",  ceil(Int,     sqrt(n))),
            ("sqrt(n)/2", max(2, ceil(Int, sqrt(n) / 2))),
        ]

        for (g_label, g) in g_cases
            println("  g=$g ($g_label) → $(g^dim) intersections")

            grid_vals = range(0, stop=L, length=g)

            if dim == 2
                all_intersections = [[grid_vals[i], grid_vals[j]] for i in 1:g for j in 1:g]
            else
                all_intersections = [
                    Float64[grid_vals[idx[k]] for k in 1:dim]
                    for idx in CartesianIndices(NTuple{dim, Base.OneTo{Int}}(fill(g, dim)))
                ]
            end

            best_obj = Inf

            for _ in 1:n_starts
                model = Model(Ipopt.Optimizer)
                set_optimizer_attribute(model, "print_level", 0)

                @variable(model, x_opt[1:n, 1:dim] >= 0)

                # Tirage AVEC remise : plusieurs points peuvent partir du même nœud
                chosen_points = sample(all_intersections, n, replace=true)
                for i in 1:n
                    for k in 1:dim
                        set_start_value(x_opt[i,k], chosen_points[i][k])
                    end
                end

                @objective(model, Min, sum(
                    ( c6 / (sum((x_opt[i,k] - x_opt[j,k])^2 for k in 1:dim))^3 - Q[i,j] )^2
                    for i in 1:n for j in i+1:n
                ))

                optimize!(model)

                obj = objective_value(model)
                if obj < best_obj
                    best_obj = obj
                end
            end

            println("  best_obj = $best_obj")
            push!(results, (n, dim, g_label, g, best_obj))
        end
    end
end

csv_path = joinpath(@__DIR__, "results_compare_g_tirage_non_unique.csv")
CSV.write(csv_path, results)
println("\nResultats sauvegardes dans : $csv_path")
println(results)
