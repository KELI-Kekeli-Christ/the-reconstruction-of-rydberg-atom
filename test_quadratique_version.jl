using JuMP
using Ipopt
using LinearAlgebra
using Statistics

n = 5
c6 = 1
L = sqrt(n)
dim = 2

# Generate initial reference points
x_ref = rand(n, dim) * L

for i in 1:n, j in i+1:n
    while norm(x_ref[i,:] - x_ref[j,:]) < 1
        x_ref[i,:] = rand(dim) * L
    end
end

# Calculate target distances (q_ij)
Q_target = zeros(Float64, n, n)
for i in 1:n, j in i+1:n
    d = norm(x_ref[i,:] - x_ref[j,:])
    Q_target[i,j] = c6 / (d^6)
    Q_target[j,i] = Q_target[i,j]
end

# Initialize Model
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0)
set_optimizer_attribute(model, "tol", 1e-12)

# Define Variables
@variable(model, 0 <= x[1:n, 1:dim] <= L)
@variable(model, r[i=1:n, j=i+1:n])
@variable(model, w[i=1:n, j=i+1:n])  
@variable(model, z[i=1:n, j=i+1:n])
@variable(model, mu[i=1:n, j=i+1:n])

# Set initial starting values
for i in 1:n, j in 1:dim
    set_start_value(x[i,j], x_ref[i,j])
end

# Define Objective
@objective(model, Min, sum( (r[i,j] - Q_target[i,j])^2 for i in 1:n, j in i+1:n ))

# ==========================================
# EXPLICIT MATRIX FORMULATION CONSTRAINTS
# ==========================================

for i in 1:n, j in i+1:n
    
    # Define the local 8x1 variable vector for this (i,j) pair
    # Indices: 1=r, 2=w, 3=mu, 4=z, 5=xi_1, 6=xi_2, 7=xj_1, 8=xj_2
    X_vec = [r[i,j], w[i,j], mu[i,j], z[i,j], x[i,1], x[i,2], x[j,1], x[j,2]]
    
    # 1. Constraint: r_ij * w_ij = C_6
    Q1 = zeros(8, 8)
    Q1[1, 2] = 1; Q1[2, 1] = 1
    b1 = zeros(8)
    @constraint(model, 0.5 * X_vec' * Q1 * X_vec + b1' * X_vec == c6)

    # 2. Constraint: mu_ij * z_ij - w_ij = 0
    Q2 = zeros(8, 8)
    Q2[3, 4] = 1; Q2[4, 3] = 1
    b2 = zeros(8)
    b2[2] = -1
    @constraint(model, 0.5 * X_vec' * Q2 * X_vec + b2' * X_vec == 0)

    # 3. Constraint: z_ij^2 - mu_ij = 0
    Q3 = zeros(8, 8)
    Q3[4, 4] = 2
    b3 = zeros(8)
    b3[3] = -1
    @constraint(model, 0.5 * X_vec' * Q3 * X_vec + b3' * X_vec == 0)

    # 4. Constraint: ||x_i - x_j||^2 - z_ij = 0
    Q4 = zeros(8, 8)
    Q4[5, 5] = 2; Q4[6, 6] = 2       # x_i squared terms
    Q4[7, 7] = 2; Q4[8, 8] = 2       # x_j squared terms
    Q4[5, 7] = -2; Q4[7, 5] = -2     # x_i1 * x_j1 cross terms
    Q4[6, 8] = -2; Q4[8, 6] = -2     # x_i2 * x_j2 cross terms
    b4 = zeros(8)
    b4[4] = -1
    @constraint(model, 0.5 * X_vec' * Q4 * X_vec + b4' * X_vec == 0)
    
end

# Optimize the model
optimize!(model)

# Check status
println("Termination Status: ", termination_status(model))
println("Objective Value: ", objective_value(model))
println("Solution x: ", value.(x))


