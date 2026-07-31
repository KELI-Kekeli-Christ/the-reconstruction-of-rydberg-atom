# ═══════════════════════════════════════════════════════════════
#  Diagnostic Taylor : pourquoi r_ij → 0 (Eryd « infinie »)
#
#  julia test_real/diag_taylor_collapse.jl
# ═══════════════════════════════════════════════════════════════

using JuMP, Ipopt, LinearAlgebra, Random, Printf, Statistics, DataFrames

const MOI = JuMP.MOI
const c6 = 1.0
const dim = 2
const z_min = 0.04
const d_min = 0.04

include(joinpath(@__DIR__, "utils.jl"))
include(joinpath(@__DIR__, "..", "formulations.jl"))
include(joinpath(@__DIR__, "formulations_Lvar.jl"))

function print_pairs(x, Q; label="", zmin=nothing, max_pairs=15)
    n = size(x, 1)
    pairs = Tuple{Int,Int,Float64,Float64,Float64,Float64}[]
    for i in 1:n, j in i+1:n
        rij = norm(x[i, :] - x[j, :])
        dij = (c6 / Q[i, j])^(1 / 6)
        w = 36 / dij^14                      # poids Taylor
        Qhat = c6 / max(rij, 1e-30)^6
        push!(pairs, (i, j, rij, dij, w, (Q[i, j] - Qhat)^2))
    end
    sort!(pairs; by=t -> t[3])  # plus petites distances d'abord
    @printf("\n  %s — %d paires (triées par r↑), affichage ≤%d\n",
            label, length(pairs), max_pairs)
    @printf("  %-8s %12s %10s %10s %12s %12s %12s\n",
            "paire", "r_ij", "d_ij", "|r-d|", "36/d^14", "C6/r^6", "(ΔQ)^2")
    for (k, (i, j, rij, dij, w, term)) in enumerate(pairs)
        k > max_pairs && break
        flag = (zmin !== nothing && rij < 0.99 * zmin) ? " ⚠<zmin" :
               (rij < 1e-4) ? " ★ COLLAPSE" : ""
        @printf("  (%d,%d)%s %12.4e %10.4f %10.3e %12.3e %12.3e %12.3e%s\n",
                i, j, "", rij, dij, abs(rij - dij), w,
                c6 / max(rij, 1e-30)^6, term, flag)
    end
    dmin = pairs[1][3]
    @printf("  → dmin=%.6e", dmin)
    if zmin !== nothing
        @printf("  zmin=%.6e  ratio=%.4f", zmin, dmin / zmin)
    end
    println()
end

function print_x(x; label="")
    n = size(x, 1)
    println("  $label — coordonnées x:")
    for i in 1:n
        @printf("    x[%2d] = (%10.6f, %10.6f)\n", i, x[i, 1], x[i, 2])
    end
    # détection de points quasi-identiques
    for i in 1:n, j in i+1:n
        rij = norm(x[i, :] - x[j, :])
        if rij < 1e-4
            @printf("    ★ points confondus : i=%d et j=%d  (‖xi-xj‖=%.3e)\n", i, j, rij)
        end
    end
end

function run_one(n; alpha=1e-4, seed=42, L_init=10.0)
    Random.seed!(seed)
    x_ref = generer_init(n, dim, L_init; seuil=z_min)
    Q = build_Q_matrix(x_ref, c6, n)
    zmin = zmin_optimal(Q, c6)
    x0 = generer_init(n, dim, L_init; seuil=z_min)

    println("\n" * "═"^72)
    @printf("  n=%d  α=%.0e  seed=%d  zmin_opt=%.6f  Qmax=%.4e\n",
            n, alpha, seed, zmin, maximum(Q))
    println("═"^72)

    println("\n▶ Référence (x généré → Q exacte)")
    print_x(x_ref; label="x_ref")
    print_pairs(x_ref, Q; label="référence", zmin=zmin, max_pairs=min(12, n*(n-1)÷2))

    for (tag, zarg) in (
        ("Taylor AVEC zmin", zmin),
        ("Taylor SANS zmin", nothing),
        ("Simple SANS zmin", nothing),  # contraste
    )
        solver = startswith(tag, "Taylor") ? solve_taylor_Lvar : solve_simple_Lvar
        res = solver(n, dim, Q, c6, x0;
                     alpha=alpha, zmin=zarg, L_init=L_init,
                     L_lb=zarg === nothing ? 1e-3 : Float64(zarg))
        if res === nothing
            println("\n▶ $tag — ÉCHEC")
            continue
        end
        status, obj, x, L = res
        ry = erreur_rydberg_vraie(x, Q, c6, n, dim)
        dmin = min_pairwise_distance(x)
        println("\n▶ $tag")
        @printf("  status=%s  L*=%.4f  dmin=%.6e  Eryd=%.3e  obj=%.3e\n",
                status, L, dmin, ry, obj)
        print_x(x; label="x*")
        print_pairs(x, Q; label="solution", zmin=zmin, max_pairs=min(12, n*(n-1)÷2))
    end
end

println("Diagnostic collapse Taylor — poids 36/d^14 trop faible sur paires lointaines")
for n in (4, 10)
    run_one(n; alpha=1e-4, seed=42)
end
println("\n" * "═"^72)
println("Conclusion : sans zmin, Taylor peut coller 2 atomes (r→0).")
println("Le poids Taylor 36/d^14 est ≈0 pour d grand → (r-d)^2 ne coûte rien,")
println("alors que Eryd = Σ(Q-C6/r^6)^2 → ∞. Simple pénalise (r-d)^2 uniformément.")
println("═"^72)
