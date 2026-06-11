using Pkg
using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

n_values  = [2,3,100]
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

        t0 = time()
        m2 = Model(Ipopt.Optimizer)
        set_optimizer_attribute(m2, "print_level", 0)
        set_optimizer_attribute(m2, "tol", 1e-12)

        @variable(m2, x2[1:n, 1:dim] >= 0)
        @variable(m2, r[i=1:n, j=i+1:n])

        # Points de départ
        for i in 1:n, d in 1:dim
            set_start_value(x2[i,d], x_init[i,d])
        end
        for i in 1:n
            for j in i+1:n
                d_sq = sum((x_init[i,k] - x_init[j,k])^2 for k in 1:dim)
                set_start_value(r[i,j], c6 / d_sq^3)
            end
        end

        # Contrainte : r[i,j] * (||xi-xj||^2)^3 = c6
        @constraint(m2, [i=1:n, j=i+1:n], 
            r[i,j] * (sum((x2[i,k] - x2[j,k])^2 for k=1:dim))^3 == c6
        )

        @objective(m2, Min, sum((r[i,j] - Q[i,j])^2 for i=1:n for j=i+1:n))

        optimize!(m2)
        t2 = time() - t0
        obj2 = objective_value(m2)
        status = termination_status(m2)
        println("  [Reformulée] status=$status  obj=$obj2  t=$(round(t2,digits=2))s")
        push!(results, (n, dim, "reformulee", obj2, string(status), t2))
    end
end

println(results)