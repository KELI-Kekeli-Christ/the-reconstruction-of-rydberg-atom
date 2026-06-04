using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

n_values  = [2,3,10]
limit_dim = 2
L         = 10
c6        = 1

const MOI = JuMP.MOI

results = DataFrame(
    n           = Int[],
    dim         = Int[],
    formulation = String[],
    obj         = Float64[],
    status      = String[],
    time_s      = Float64[]
)

for n in n_values
    for dim in 2:limit_dim
        println("\n=== n=$n, dim=$dim ===")

        x_init = L * rand(n, dim)
        Q = zeros(n, n)
        for i in 1:n
            for j in i+1:n
                d = norm(x_init[i,:] - x_init[j,:])
                Q[i,j] = c6 / d^6
                Q[j,i] = c6 / d^6
            end
        end

        # ================================================================
        # FORMULATION STANDARD  — objectif direct, pas de contraintes
        # min sum_{i<j} ( c6/||xi-xj||^6 - Q[i,j] )^2
        # ================================================================
        t0 = time()
        m1 = Model(Ipopt.Optimizer)
        set_optimizer_attribute(m1, "print_level", 0)
        set_optimizer_attribute(m1, "tol", 1e-12)

        @variable(m1, x1[1:n, 1:dim] >= 0)
        for i in 1:n, d in 1:dim
            set_start_value(x1[i,d], x_init[i,d])
        end

        @objective(m1, Min, sum(
            ( c6 / (sum((x1[i,k] - x1[j,k])^2 for k in 1:dim))^3 - Q[i,j] )^2
            for i in 1:n for j in i+1:n
        ))

        optimize!(m1)
        t1 = time() - t0
        obj1 = objective_value(m1)
        println("  [Standard]   status=$(termination_status(m1))  obj=$obj1  t=$(round(t1,digits=2))s")
        push!(results, (n, dim, "standard", obj1, string(termination_status(m1)), t1))

        # ================================================================
        # FORMULATION REFORMULÉE  — variables auxiliaires + contraintes
        # Variables : x, w[i,j], u[i,j], r[i,j]
        #   w[i,j] = sum_k (xi_k - xj_k)^2          (distance²)
        #   u[i,j] = w[i,j]^3                         (distance^6)
        #   r[i,j] * u[i,j] = c6                      (r = c6/d^6)
        # Objectif : min sum_{i<j} ( r[i,j] - Q[i,j] )^2
        # ================================================================
        t0 = time()
        m2 = Model(Ipopt.Optimizer)
        set_optimizer_attribute(m2, "print_level", 0)
        set_optimizer_attribute(m2, "tol", 1e-12)
     

        @variable(m2, x2[1:n, 1:dim]>=0 )
        @variable(m2, r[i=1:n, j=(i+1):n] )   # c6 / d^6
        @variable(m2, w[i=1:n, j=(i+1):n]  )   # d^6  (= u*z)
        @variable(m2, u[i=1:n, j=(i+1):n] )   # d^4  (= z^2)
        @variable(m2, z[i=1:n, j=(i+1):n] )   # d^2

        # Points de départ
        for i in 1:n, d in 1:dim
            set_start_value(x2[i,d], x_init[i,d])
        end
        for i in 1:n
            for j in i+1:n
                d_sq = sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim)
                z_val = d_sq
                u_val = d_sq^2
                w_val = d_sq^3
                set_start_value(z[i,j], z_val)
                set_start_value(u[i,j], u_val)
                set_start_value(w[i,j], w_val)
                set_start_value(r[i,j], c6 / w_val)
            end
        end

        # Contrainte 1 : r[i,j] * w[i,j] == c6
        @constraint(m2, [i=1:n, j=(i+1):n], r[i,j] * w[i,j] == c6)
        # Contrainte 2 : z[i,j] == ||xi - xj||²
        @constraint(m2, [i=1:n, j=(i+1):n], z[i,j] == sum((x2[i,k] - x2[j,k])^2 for k in 1:dim))
        # Contrainte 3 : w[i,j] == u[i,j] * z[i,j]
        @constraint(m2, [i=1:n, j=(i+1):n], w[i,j] == u[i,j] * z[i,j])
        # Contrainte 4 : u[i,j] == z[i,j]^2
        @constraint(m2, [i=1:n, j=(i+1):n], u[i,j] == z[i,j]^2)

        @objective(m2, Min, sum((r[i,j] - Q[i,j])^2 for i in 1:n for j in i+1:n))

        optimize!(m2)
        t2 = time() - t0
        obj2 = objective_value(m2)
        println("  [Reformulée] status=$(termination_status(m2))  obj=$obj2  t=$(round(t2,digits=2))s")
        push!(results, (n, dim, "reformulee", obj2, string(termination_status(m2)), t2))
    end
end

csv_path = joinpath(@__DIR__, "results_formulation_vs.csv")
CSV.write(csv_path, results)
println("\nResultats sauvegardes dans : $csv_path")
println(results)
