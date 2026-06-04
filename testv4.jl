using JuMP, Ipopt, LinearAlgebra, CSV, DataFrames

const C6 = 862690.0
const L  = 10.0
limit_points = 20
n_starts     = 100
results = []

for n in 3:limit_points
    x_true = L .* rand(n, 2)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        r2 = sum((x_true[i,k] - x_true[j,k])^2 for k in 1:2)
        Q[i,j] = C6 / r2^3
        Q[j,i] = Q[i,j]
    end

    log_q = zeros(n, n)
    for i in 1:n, j in i+1:n
        log_q[i,j] = log(Q[i,j])
        log_q[j,i] = log_q[i,j]
    end

    for _ in 1:n_starts
        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, x[1:n, 1:2])

        x_init = L .* rand(n, 2)
        for i in 1:n, k in 1:2
            set_start_value(x[i,k], x_init[i,k])
        end

        @objective(model, Min,
            sum((log(C6) - 3*log((x[i,1]-x[j,1])^2 +
                                  (x[i,2]-x[j,2])^2)
                 - log_q[i,j])^2
                for i in 1:n, j in i+1:n))

        optimize!(model)

        obj = objective_value(model)
        println("n=$n | obj=$obj | status=$(termination_status(model))")
        push!(results, (n=n, obj=obj, status=string(termination_status(model))))
    end
end

#CSV.write("resultats_avec_log.csv", DataFrame(results))