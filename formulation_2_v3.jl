using JuMP, SCIP, LinearAlgebra, CSV, DataFrames
using Statistics

const C6 = 862690.0
limit_points = 12          
results = []

for n in 3:limit_points
    for dim in 2:2
        Q = zeros(n, n)
        for i in 1:n, j in i+1:n
            Q[i,j] = Q[j,i] = abs(randn()) * 10.0
        end
        
        q_mean = sum(Q[i,j] for i in 1:n for j in i+1:n) / (n*(n-1)/2)
        R_typ = (C6 / q_mean)^(1/6)
        L_init = R_typ * sqrt(n) * 1.3

        model = Model(SCIP.Optimizer)

        #set_attribute(model, "limits/time", 300.0)      # 5 minutes
        set_attribute(model, "limits/gap", 1e-4)
        set_attribute(model, "display/verblevel", 2)

        @variable(model, x[1:n, 1:dim])
        @variable(model, r2[i=1:n, j=i+1:n] >= 1e-6)
        @variable(model, r4[i=1:n, j=i+1:n] >= 1e-6)   # r2²
        @variable(model, r6[i=1:n, j=i+1:n] >= 1e-6)   # r4 * r2
        @variable(model, y[i=1:n, j=i+1:n])

        # Initialisation
        x_init = L_init .* rand(n, dim)
        set_start_value.(x, x_init)

        for i in 1:n, j in i+1:n
            dx = x_init[i,1] - x_init[j,1]
            dy = x_init[i,2] - x_init[j,2]
            dist2 = max(dx^2 + dy^2, 1e-5)
            set_start_value(r2[i,j], dist2)
            set_start_value(r4[i,j], dist2^2)
            set_start_value(r6[i,j], dist2^3)
            set_start_value(y[i,j], C6 / dist2^3)
        end

        @objective(model, Min, sum((y[i,j] - Q[i,j])^2 for i in 1:n for j in i+1:n))

        # Contraintes
        for i in 1:n, j in i+1:n
            @constraint(model, r2[i,j] == sum((x[i,k] - x[j,k])^2 for k in 1:dim))   # quadratic
            @constraint(model, r4[i,j] == r2[i,j] * r2[i,j])                         # quadratic
            @constraint(model, r6[i,j] == r4[i,j] * r2[i,j])                         # bilinear
            @constraint(model, y[i,j] * r6[i,j] == C6)                               # bilinear
        end

        optimize!(model)

        status = termination_status(model)
        obj = objective_value(model)

        println("n=$n | status=$status | obj=$(round(obj, digits=4)) | time=$(round(solve_time(model), digits=1))s")
        push!(results, (n=n, dim=dim, obj=obj, status=string(status)))
    end
end

df = DataFrame(results)
CSV.write("resultats_global_scip_v3.csv", df)
