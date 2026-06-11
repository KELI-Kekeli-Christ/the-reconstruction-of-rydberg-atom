using LinearAlgebra, JuMP, Ipopt, Random, Printf

Random.seed!(42)

# ═══════════════════════════════════════════════════════════════
#  §1  PARAMÈTRES
# ═══════════════════════════════════════════════════════════════
n   = 20
dim = 2
L   = 10.0
c6  = 1.0
ε   = 1e-3

# ═══════════════════════════════════════════════════════════════
#  §2  DONNÉES CIBLES
# ═══════════════════════════════════════════════════════════════
function gen_valid(n, dim, L, ε=1e-3)
    x = L * rand(n, dim)
    for i in 1:n
        while norm(x[i,:]) < ε
            x[i,:] = L * rand(dim)
        end
        for j in 1:i-1
            t = 0
            while norm(x[i,:] - x[j,:]) < ε
                x[i,:] .+= ε * rand(dim)
                x[i,:]   = clamp.(x[i,:], 0.0, L)
                t += 1
                if t > 100; x[i,:] = L * rand(dim); t = 0; end
            end
        end
    end
    return x
end

x_vrais = gen_valid(n, dim, L)
P     = n * (n-1) ÷ 2
pairs = [(i,j) for i in 1:n for j in i+1:n]

Q_mat = zeros(n, n)
for (k, (i,j)) in enumerate(pairs)
    d = max(norm(x_vrais[i,:] - x_vrais[j,:]), 1e-5)
    Q_mat[i,j] = Q_mat[j,i] = c6 / d^6
end

# ═══════════════════════════════════════════════════════════════
#  §3  VECTEUR DE VARIABLES
# ═══════════════════════════════════════════════════════════════
dim_X = 4P + 2n
@assert dim_X == 2n^2

w_id(k)    = k
z_id(k)    = P  + k
r_id(k)    = 2P + k
μ_id(k)    = 3P + k
xd_id(i,d) = 4P + (d-1)*n + i

println("━"^62)
println("  STRUCTURE DU VECTEUR X")
println("━"^62)
println("  n=$n, dim=$dim, P=n(n-1)/2=$P")
@printf("  dim_X = 4P + 2n = %d  (vérifié = 2n² = %d)\n", dim_X, 2n^2)
println("  Blocs : w[$P] | z[$P] | r[$P] | μ[$P] | x¹[$n] | x²[$n]")

# ─── Bornes physiques sur X ───────────────────────────────────
X_lo = zeros(dim_X)
X_hi = zeros(dim_X)
zmax = dim * L^2

# CORRECTION BUG 2 : bornes sur r basées sur Q_mat, pas sur ε
# r_max = c6/ε^6 = 1e18 rendait les bornes Y astronomiques
# → la contrainte relaxée devenait trivialement satisfaite
Q_vals   = [Q_mat[i,j] for (i,j) in pairs]
r_min_phys = minimum(Q_vals) * 0.001   # 3 ordres sous le Q minimum
r_max_phys = maximum(Q_vals) * 1000    # 3 ordres au-dessus du Q maximum

for k in 1:P
    X_hi[w_id(k)] = zmax^3
    X_hi[z_id(k)] = zmax
    X_lo[r_id(k)] = r_min_phys         # ← était c6/zmax^3 ≈ 10^-21
    X_hi[r_id(k)] = r_max_phys         # ← était c6/ε^6   ≈ 10^+18  !!!
    X_hi[μ_id(k)] = zmax^2
end
for i in 1:n, d in 1:dim
    X_hi[xd_id(i,d)] = L
end

# ═══════════════════════════════════════════════════════════════
#  §4  MATRICE OBJECTIF M
# ═══════════════════════════════════════════════════════════════
M     = zeros(dim_X, dim_X)
d_obj = zeros(dim_X)
cte   = sum(Q_mat[i,j]^2 for (i,j) in pairs)

for (k, (i,j)) in enumerate(pairs)
    M[r_id(k), r_id(k)] = 2.0
    d_obj[r_id(k)]       = -2.0 * Q_mat[i,j]
end

# ═══════════════════════════════════════════════════════════════
#  §5  MATRICE DE CONTRAINTE Q₂
# ═══════════════════════════════════════════════════════════════
Q2    = zeros(dim_X, dim_X)
b_con = zeros(dim_X)
c_con = Float64(P) * c6

for (k, (i,j)) in enumerate(pairs)
    Q2[r_id(k), w_id(k)] += 1.0
    Q2[w_id(k), r_id(k)] += 1.0

    for d in 1:dim
        a, b = xd_id(i,d), xd_id(j,d)
        Q2[a, a] += 2.0;  Q2[b, b] += 2.0
        Q2[a, b] -= 2.0;  Q2[b, a] -= 2.0
    end
    b_con[z_id(k)] -= 1.0

    Q2[μ_id(k), z_id(k)] += 1.0
    Q2[z_id(k), μ_id(k)] += 1.0
    b_con[w_id(k)] -= 1.0

    Q2[z_id(k), z_id(k)] += 2.0
    b_con[μ_id(k)] -= 1.0
end

# ═══════════════════════════════════════════════════════════════
#  §6  DIAGONALISATION
# ═══════════════════════════════════════════════════════════════
F = eigen(Symmetric(Q2))
λ = F.values
V = F.vectors

b_tilde = V' * b_con
M_tilde = V' * M * V
d_tilde = V' * d_obj

# ═══════════════════════════════════════════════════════════════
#  §7  SPECTRE
# ═══════════════════════════════════════════════════════════════
tol_eig = 1e-8
pos = findall(λ .>  tol_eig)
neg = findall(λ .< -tol_eig)
nul = findall(abs.(λ) .<= tol_eig)

println("\n" * "━"^62)
println("  §7  SPECTRE DE Q₂")
println("━"^62)
@printf("  λᵢ positives  : %3d  | max λ⁺ = %+.4e\n",
        length(pos), isempty(pos) ? 0.0 : maximum(λ[pos]))
@printf("  λᵢ négatives  : %3d  | min λ⁻ = %+.4e\n",
        length(neg), isempty(neg) ? 0.0 : minimum(λ[neg]))
@printf("  λᵢ ≈ 0        : %3d\n", length(nul))

# ═══════════════════════════════════════════════════════════════
#  §8  BORNES SUR Y = Vᵀ X
# ═══════════════════════════════════════════════════════════════
Y_lo = zeros(dim_X)
Y_hi = zeros(dim_X)

for i in 1:dim_X
    for k in 1:dim_X
        v = V[k, i]
        if v >= 0
            Y_lo[i] += v * X_lo[k]
            Y_hi[i] += v * X_hi[k]
        else
            Y_lo[i] += v * X_hi[k]
            Y_hi[i] += v * X_lo[k]
        end
    end
end

@printf("  ||Y_lo||∞ = %.3e  ||Y_hi||∞ = %.3e\n",
        maximum(abs.(Y_lo)), maximum(abs.(Y_hi)))

# ═══════════════════════════════════════════════════════════════
#  §9  RELAXATION CONVEXE (bornes lâches)
# ═══════════════════════════════════════════════════════════════
mdl_rlx = Model(Ipopt.Optimizer)
set_optimizer_attribute(mdl_rlx, "print_level", 0)
set_optimizer_attribute(mdl_rlx, "tol",         1e-10)

@variable(mdl_rlx, Y_lo[i] <= y[i=1:dim_X] <= Y_hi[i])

expr_pos = isempty(pos) ? 0.0 :
    sum(0.5 * λ[i] * y[i]^2 for i in pos)

expr_lin = isempty(neg) ? 0.0 :
    sum(0.5 * λ[i] * ((Y_lo[i] + Y_hi[i]) * y[i] - Y_lo[i] * Y_hi[i])
        for i in neg)

expr_b = sum(b_tilde[i] * y[i] for i in 1:dim_X)

@constraint(mdl_rlx, expr_pos + expr_lin + expr_b <= c_con)

@objective(mdl_rlx, Min,
    sum(0.5 * M_tilde[i,i] * y[i]^2 for i in 1:dim_X)
    + sum(M_tilde[i,j] * y[i] * y[j] for i in 1:dim_X for j in i+1:dim_X)
    + sum(d_tilde[i] * y[i] for i in 1:dim_X)
    + cte
)

t_rlx    = @elapsed optimize!(mdl_rlx)
obj_rlx  = objective_value(mdl_rlx)
stat_rlx = termination_status(mdl_rlx)

# ═══════════════════════════════════════════════════════════════
#  §10  NLP ORIGINAL
#
#  CORRECTION BUG 1 : le NLP doit avoir x ∈ [0, L] (pas seulement x ≥ 0)
#  Sans borne supérieure, le NLP peut trouver x > L, qui est hors du
#  domaine de la relaxation → violation de la propriété borne inférieure.
# ═══════════════════════════════════════════════════════════════
x_init = gen_valid(n, dim, L)

mdl_nlp = Model(Ipopt.Optimizer)
set_optimizer_attribute(mdl_nlp, "print_level", 0)
set_optimizer_attribute(mdl_nlp, "tol",         1e-12)
set_optimizer_attribute(mdl_nlp, "max_iter",     3000)

# CORRECTION : 0 <= x_nlp <= L  (cohérent avec les bornes de la relaxation)
@variable(mdl_nlp, 0 <= x_nlp[1:n, 1:dim] <= L)

for i in 1:n, d in 1:dim
    set_start_value(x_nlp[i,d], x_init[i,d])
end

@NLobjective(mdl_nlp, Min,
    sum(
        (c6 / (sum((x_nlp[i,k] - x_nlp[j,k])^2 for k in 1:dim))^3 - Q_mat[i,j])^2
        for (i,j) in pairs
    )
)

t_nlp    = @elapsed optimize!(mdl_nlp)
obj_nlp  = objective_value(mdl_nlp)
stat_nlp = termination_status(mdl_nlp)

# ═══════════════════════════════════════════════════════════════
#  §11  RELAXATION SERRÉE autour de Y*
#
#  CORRECTION BUG 3 : δ adaptatif proportionnel à ||Y*||
#  δ = 0.5 était trop petit pour n=10 (||Y*|| ~ O(1e3))
#  → les bornes serrées excluaient Y* → ITERATION_LIMIT
# ═══════════════════════════════════════════════════════════════
x_nlp_sol = value.(x_nlp)

# Reconstruction de X* depuis la solution NLP
X_star = zeros(dim_X)
for (k, (i,j)) in enumerate(pairs)
    dij = x_nlp_sol[i,:] - x_nlp_sol[j,:]
    zij = max(sum(dij.^2), 1e-10)
    X_star[w_id(k)] = zij^3
    X_star[z_id(k)] = zij
    X_star[r_id(k)] = c6 / zij^3
    X_star[μ_id(k)] = zij^2
end
for i in 1:n, d in 1:dim
    X_star[xd_id(i,d)] = x_nlp_sol[i,d]
end

Y_star = V' * X_star

# δ adaptatif : 20% de la norme de Y*, au moins 1.0
Y_norm = max(norm(Y_star), 1.0)
δ = Y_norm * 0.2
@printf("  ||Y*|| = %.3e  →  δ adaptatif = %.3e\n", Y_norm, δ)

Y_lo_tight = max.(Y_lo, Y_star .- δ)
Y_hi_tight = min.(Y_hi, Y_star .+ δ)

# Sécurité : garantir Y_lo_tight ≤ Y_hi_tight
for i in 1:dim_X
    if Y_lo_tight[i] > Y_hi_tight[i]
        mid = (Y_lo[i] + Y_hi[i]) / 2.0
        Y_lo_tight[i] = mid - δ
        Y_hi_tight[i] = mid + δ
    end
end

mdl_tight = Model(Ipopt.Optimizer)
set_optimizer_attribute(mdl_tight, "print_level", 0)
set_optimizer_attribute(mdl_tight, "tol",         1e-10)
set_optimizer_attribute(mdl_tight, "max_iter",     5000)

@variable(mdl_tight, Y_lo_tight[i] <= y2[i=1:dim_X] <= Y_hi_tight[i])

expr_pos2 = isempty(pos) ? 0.0 :
    sum(0.5 * λ[i] * y2[i]^2 for i in pos)

expr_lin2 = isempty(neg) ? 0.0 :
    sum(0.5 * λ[i] * ((Y_lo_tight[i] + Y_hi_tight[i]) * y2[i]
                      - Y_lo_tight[i] * Y_hi_tight[i])
        for i in neg)

expr_b2 = sum(b_tilde[i] * y2[i] for i in 1:dim_X)

@constraint(mdl_tight, expr_pos2 + expr_lin2 + expr_b2 <= c_con)

@objective(mdl_tight, Min,
    sum(0.5 * M_tilde[i,i] * y2[i]^2 for i in 1:dim_X)
    + sum(M_tilde[i,j] * y2[i] * y2[j] for i in 1:dim_X for j in i+1:dim_X)
    + sum(d_tilde[i] * y2[i] for i in 1:dim_X)
    + cte
)

t_tight    = @elapsed optimize!(mdl_tight)
obj_tight  = objective_value(mdl_tight)
stat_tight = termination_status(mdl_tight)

# ═══════════════════════════════════════════════════════════════
#  §12  TABLEAU COMPARATIF FINAL
# ═══════════════════════════════════════════════════════════════
gap_lache = obj_nlp - obj_rlx
gap_serre = obj_nlp - obj_tight

println("\n" * "━"^62)
println("  §12  TABLEAU COMPARATIF FINAL")
println("━"^62)
println()
@printf("  %-32s : %+.6e   [%s]\n", "NLP original (borne sup)",
        obj_nlp, stat_nlp)
@printf("  %-32s : %+.6e   [%s]\n", "Relaxation lâche (borne inf)",
        obj_rlx, stat_rlx)
@printf("  %-32s : %+.6e   [%s]\n", "Relaxation serrée (borne inf)",
        obj_tight, stat_tight)
println()
@printf("  Gap lâche   (NLP − relax_lâche)   : %+.6e\n", gap_lache)
@printf("  Gap serré   (NLP − relax_serrée)  : %+.6e\n", gap_serre)
println()

# Vérification de la propriété de borne inférieure
println("  Vérification propriété borne inférieure :")
ok_lache  = obj_rlx  <= obj_nlp + 1e-8
ok_serre  = obj_tight <= obj_nlp + 1e-8
@printf("  relax_lâche  ≤ NLP : %s  (%.3e ≤ %.3e)\n",
        ok_lache  ? "✓" : "✗", obj_rlx,   obj_nlp)
@printf("  relax_serrée ≤ NLP : %s  (%.3e ≤ %.3e)\n",
        ok_serre  ? "✓" : "✗", obj_tight, obj_nlp)
println()

if ok_serre && ok_lache
    amelioration = abs(gap_lache) > 1e-30 ?
        (abs(gap_lache) - abs(gap_serre)) / abs(gap_lache) * 100 : 0.0
    @printf("  Resserrement : gap réduit de %.2f%%\n", amelioration)
    println("  → La relaxation serrée est une meilleure borne inférieure.")
else
    println("  ⚠ Propriété violée : vérifier cohérence des domaines.")
end

println()
println("  ┌───────────────────────────────────────────────────────┐")
println("  │  gap → 0 : relaxation exacte (idéal)                │")
println("  │  gap grand : renforcer les bornes ou branch-and-bound │")
println("  └───────────────────────────────────────────────────────┘")
println()
@printf("  Temps NLP             : %.4f s\n", t_nlp)
@printf("  Temps relax lâche     : %.4f s\n", t_rlx)
@printf("  Temps relax serrée    : %.4f s\n", t_tight)
println("━"^62)