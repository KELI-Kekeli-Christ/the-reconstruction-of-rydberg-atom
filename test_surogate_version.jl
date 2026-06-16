using JuMP, Ipopt, LinearAlgebra,Statistics

n = 10
c6 = 1
L = sqrt(n)
dim = 2

x_ref = rand(n, dim) * L

for i in 1:n, j in i+1:n
    while norm(x_ref[i,:] - x_ref[j,:]) < 1
        x_ref[i,:] = rand(dim) * L
    end
end

Q = zeros(Float64, n, n)
for i in 1:n, j in i+1:n
    d = norm(x_ref[i,:] - x_ref[j,:])
    Q[i,j] = c6 / (d^6)
    Q[j,i] = Q[i,j]
end

model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0)
set_optimizer_attribute(model, "tol", 1e-12)
@variable(model, 0 <= x[1:n, 1:dim] <= L)
@variable(model,r[i=1:n, j=i+1:n])
@variable(model,w[i=1:n, j=i+1:n])  
@variable(model,z[i=1:n, j=i+1:n])
@variable(model, mu[i=1:n, j=i+1:n])

for i in 1:n, j in 1:dim
    set_start_value(x[i,j], x_ref[i,j])
end

@objective(model,Min, sum((r[i,j]-Q[i,j] for i in 1:n for j in i+1:n).^2))

@constraint(model , [i=1:n, j=i+1:n],r[i,j]*w[i,j]+z[i,j]+w[i,j]+mu[i,j]-mu[i,j]*z[i,j]-z[i,j]^2-(x[i,1]-x[j,1])^2-(x[i,2]-x[j,2])^2-c6==0)

optimize!(model)

obj=objective_value(model)
println("Objective value: ", obj)
println("Termination status: ", termination_status(model))


function rmse_dist(x, x_ref, n)
    err = 0.0
    count = 0

    for i in 1:n, j in i+1:n
        d1 = norm(x[i,:] - x[j,:])
        d2 = norm(x_ref[i,:] - x_ref[j,:])
        err += (d1 - d2)^2
        count += 1
    end

    return sqrt(err / count)
end

x_sol = value.(x)

println("RMSE distances = ", rmse_dist(x_sol, x_ref, n))