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

    for _ in 1:n_starts
        model = Model(Ipopt.Optimizer)
        set_silent(model)

        @variable(model, x[1:n, 1:2])
        @variable(model, a[i=1:n, j=i+1:n])
        @variable(model, b[i=1:n, j=i+1:n])
        @variable(model, y[i=1:n, j=i+1:n])

        x_init = L .* rand(n, 2)
        for i in 1:n, k in 1:2
            set_start_value(x[i,k], x_init[i,k])
        end
        for i in 1:n, j in i+1:n
            ai = x_init[i,1] - x_init[j,1]
            bi = x_init[i,2] - x_init[j,2]
            if abs(ai) < 1e-3 && abs(bi) < 1e-3
                ai = 1.0; bi = 1.0
            end
            set_start_value(a[i,j], ai)
            set_start_value(b[i,j], bi)
            set_start_value(y[i,j], C6 / (ai^2 + bi^2)^3)
        end

        @objective(model, Min,
            sum((y[i,j] - Q[i,j])^2 for i in 1:n, j in i+1:n))

        @constraint(model, [i in 1:n, j in i+1:n],
            a[i,j] == x[i,1] - x[j,1])
        @constraint(model, [i in 1:n, j in i+1:n],
            b[i,j] == x[i,2] - x[j,2])
        @constraint(model, [i in 1:n, j in i+1:n],
            y[i,j] * (a[i,j]^6 + 3*a[i,j]^4*b[i,j]^2
                    + 3*a[i,j]^2*b[i,j]^4 + b[i,j]^6) == C6)

        optimize!(model)

        obj = objective_value(model)
        println("n=$n | obj=$obj | status=$(termination_status(model))")
        push!(results, (n=n, obj=obj, status=string(termination_status(model))))
    end
end

#CSV.write("resultats_sans_log.csv", DataFrame(results))