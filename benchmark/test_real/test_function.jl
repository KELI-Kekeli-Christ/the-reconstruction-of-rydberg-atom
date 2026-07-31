using JuMP, Ipopt, LinearAlgebra, DataFrames, Random, Printf, CSV, Statistics

const MOI = JuMP.MOI
const z_min = 0.0016
const d_min = 0.04
const c6 = 1.0
const dim = 2

include("utils.jl")
include(joinpath(@__DIR__, "..", "formulations.jl"))

const PASQAL_ROOT = joinpath(@__DIR__, "..", "pasqal")
const PASQAL_FOLDERS = ["sqauare", "freespace", "triangle"]
const ALL_L_TYPES = ["L_fixe", "L_adaptatif", "L_optimal"]

const ALL_FORMULATIONS = Pair{String, Function}[
    "Standard"      => solve_standard,
    # "new_standard"  => solve_new_standard,
    # "Reformulee"    => solve_reformulee,
    # "Polaire"       => solve_polaire,
    # "Surogate"      => solve_surogate,
    # "Quadratique"   => solve_quadratique,
    "Taylor"        => (n, dim, Q, c6, L, x) -> formulation_taylor(Q, n, dim, L, c6, x),
    "Taylor_square" => (n, dim, Q, c6, L, x) -> formulation_taylor_square(Q, n, dim, L, c6, x),
    "Simple_square" => (n, dim, Q, c6, L, x) -> formulation_simple_square(Q, n, dim, L, c6, x),
    "Simple"        => (n, dim, Q, c6, L, x) -> formulation_simple(Q, n, dim, L, c6, x),
]

function nb_restarts_bench(n::Int)
    n <= 10 ? 5 : n <= 20 ? 3 : 2
end

function pasqal_gml_paths()
    instances = NamedTuple[]
    for folder in PASQAL_FOLDERS
        folder_path = joinpath(PASQAL_ROOT, folder)
        isdir(folder_path) || (println("Dossier absent : $folder_path"); continue)
        for f in sort(readdir(folder_path))
            endswith(lowercase(f), ".gml") || continue
            push!(instances, (
                folder = folder,
                name = f,
                path = joinpath(folder_path, f),
            ))
        end
    end
    return instances
end

function run_formulation(solver_fn, n, dim, Q, c6, L, x_inits_shared, multistart::Bool)
    t0 = time()
    best_obj = Inf
    best_ry = Inf
    best_status = "ECHEC"
    best_x = nothing
    n_runs = multistart ? length(x_inits_shared) : 1
    n_used = 0

    for k in 1:n_runs
        res = solver_fn(n, dim, Q, c6, L, x_inits_shared[k])
        n_used += 1
        if res !== nothing
            stat, obj, x_sol = res
            ry = erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
            if ry < best_ry
                best_ry = ry
                best_obj = obj
                best_status = string(stat)
                best_x = x_sol
            end
            best_ry < 1e-12 && break
        end
    end

    return (best_obj, best_ry, best_status, best_x, n_used, time() - t0)
end

function run_mds(n, dim, Q, c6)
    t0 = time()
    try
        res = solve_mds(n, dim, Q, c6)
        res === nothing && return (NaN, Inf, "ECHEC", nothing, 0, time() - t0)
        _, x_mds = res
        ry = erreur_rydberg_vraie(x_mds, Q, c6, n, dim)
        return (0.0, ry, "MDS_EXACT", x_mds, 1, time() - t0)
    catch err
        return (NaN, Inf, "ERREUR: $err", nothing, 0, time() - t0)
    end
end

function mds_initial_point(n, dim, Q, c6, L)
    try
        res = solve_mds(n, dim, Q, c6)
        res === nothing && return generer_init(n, dim, L)
        _, x_mds = res
        x = copy(x_mds)
        for d in 1:dim
            mn, mx = extrema(x[:, d])
            span = mx - mn
            span <= 0 && continue
            x[:, d] .= (x[:, d] .- mn) ./ span .* L
        end
        return x
    catch
        return generer_init(n, dim, L)
    end
end

function run_one_gml(gml_path::AbstractString, folder::AbstractString, name::AbstractString;
                     multistart::Bool=true, L_types::Vector{String}=ALL_L_TYPES)
    Q = read_gml_q(gml_path)
    n = size(Q, 1)
    n_edges = length(active_pairs(Q))
    n_edges == 0 && error("Aucune arête pondérée dans $gml_path")

    results = DataFrame(
        folder = String[],
        file = String[],
        n = Int[],
        n_edges = Int[],
        L_type = String[],
        L_val = Float64[],
        formulation = String[],
        obj_solver = Float64[],
        err_rydberg = Float64[],
        time_s = Float64[],
        status = String[],
        n_restarts = Int[],
    )

    println("\n" * "═" ^ 70)
    println("  $folder / $name")
    @printf("  n = %d, arêtes actives = %d\n", n, n_edges)
    println("═" ^ 70)

    for L_type in L_types
        L = L_value(L_type, n; Q=Q, c6=c6)
        n_rs = multistart ? nb_restarts_bench(n) : 1
        x_mds = mds_initial_point(n, dim, Q, c6, L)
        x_inits = [x_mds]
        for _ in 2:n_rs
            push!(x_inits, generer_init(n, dim, L))
        end

        println("\n--- $L_type  (L = $(round(L, digits=4))) ---")

        obj_m, ry_m, st_m, _, nr_m, t_m = run_mds(n, dim, Q, c6)
        @printf("  %-14s  err_rydberg = %+.4e  t = %.4fs  [%s]\n",
                "MDS_Exact", ry_m, t_m, st_m)
        push!(results, (folder, name, n, n_edges, L_type, L, "MDS_Exact",
                        obj_m, ry_m, t_m, st_m, nr_m))

        for (form_name, solver_fn) in ALL_FORMULATIONS
            try
                obj, ry, status, _, n_used, elapsed = run_formulation(
                    solver_fn, n, dim, Q, c6, L, x_inits, multistart
                )
                @printf("  %-14s  err_rydberg = %+.4e  t = %.4fs  [%s]\n",
                        form_name, ry, elapsed, status[1:min(end, 30)])
                push!(results, (folder, name, n, n_edges, L_type, L, form_name,
                                obj, ry, elapsed, status, n_used))
            catch err
                @printf("  %-14s  ERREUR : %s\n", form_name, err)
                push!(results, (folder, name, n, n_edges, L_type, L, form_name,
                                NaN, Inf, 0.0, "ERREUR: $err", 0))
            end
        end
    end

    return results
end

function run_all_pasqal_gml_tests(;
        multistart::Bool=true,
        L_types::Vector{String}=ALL_L_TYPES,
        output_csv::AbstractString=joinpath(@__DIR__, "pasqal_gml_benchmark.csv"))
    instances = pasqal_gml_paths()
    isempty(instances) && error("Aucun fichier .gml trouvé sous $PASQAL_ROOT")

    println("Benchmark Pasqal : $(length(instances)) fichiers .gml")
    println("Formulations : $(join(first.(ALL_FORMULATIONS), ", ")), MDS_Exact")
    println("Régimes L : $(join(L_types, ", "))")

    df_all = DataFrame()
    for (i, inst) in enumerate(instances)
        println("\n[$i/$(length(instances))]")
        df_inst = run_one_gml(inst.path, inst.folder, inst.name;
                              multistart=multistart, L_types=L_types)
        df_all = vcat(df_all, df_inst; cols=:union)
    end

    CSV.write(output_csv, df_all)
    println("\n" * "═" ^ 70)
    println("  Résultats exportés → $output_csv")
    println("═" ^ 70)

    println("\n── Meilleure err_rydberg par fichier (toutes méthodes) ──")
    for sub in groupby(df_all, [:folder, :file, :L_type])
        r = sub[argmin(replace(sub.err_rydberg, Inf => 1e300)), :]
        @printf("  %-12s %-45s %-12s  %s  err=%.4e\n",
                r.folder, r.file, r.L_type, r.formulation, r.err_rydberg)
    end

    return df_all
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_pasqal_gml_tests()
end
