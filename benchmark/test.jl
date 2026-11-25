using JuMP, Ipopt, Random, LinearAlgebra, DataFrames, Statistics, Printf

const z_min = 0.04
const d_min = sqrt(z_min)
const c6 = 1.0
const dim = 2

include("utils.jl")
include("formulations.jl")

const PASQAL_ROOT = joinpath(@__DIR__, "pasqal")
const PASQAL_FOLDERS = ["sqauare", "freespace", "triangle"]

function pasqal_instance_paths(folder::AbstractString)
    folder_path = joinpath(PASQAL_ROOT, folder)
    txt_files = sort(filter(f -> endswith(lowercase(f), ".txt"), readdir(folder_path; join=true)))
    gml_by_base = Dict(splitext(basename(f))[1] => f for f in filter(f -> endswith(lowercase(f), ".gml"), readdir(folder_path; join=true)))
    instances = NamedTuple[]

    for txt_path in txt_files
        base = splitext(basename(txt_path))[1]
        gml_path = get(gml_by_base, base, haskey(gml_by_base, "graph") ? gml_by_base["graph"] : nothing)
        push!(instances, (txt_path=txt_path, gml_path=gml_path))
    end

    return instances
end

function run_one_pasqal_instance(txt_path::AbstractString, gml_path, folder_name::AbstractString)
    # Charger uniquement le .txt (on ignore volontairement le .gml)
    inst = load_pasqal_instance(txt_path; gml_path=nothing, scale=1)
    # Toujours générer un point de départ aléatoire (pas de positions issues du .gml)
    x0 = generer_init(inst.n, dim, inst.L)

    println("\n══════════════════════════════════════════════════════")
    println("Dossier: $folder_name")
    println("Fichier: $(basename(txt_path))")
    println("n = $(inst.n), arêtes = $(length(active_pairs(inst.Q)))")
    println("══════════════════════════════════════════════════════")

    s_mds = solve_mds(inst.n, dim, inst.Q, c6)
    if s_mds !== nothing
        _, x_mds = s_mds
        ry_mds = erreur_rydberg_vraie(x_mds, inst.Q, c6, inst.n, dim)
        println(@sprintf("[MDS_Exact]   err_rydberg = %.6e", ry_mds))
    else
        println("[MDS_Exact] impossible")
    end

    # tester plusieurs valeurs de L
    for L_type in ["L_fixe", "L_adaptatif", "L_optimal"]
        L_val = try
            L_value(L_type, inst.n; Q=inst.Q, c6=c6)
        catch err
            println("Impossible de calculer L pour $L_type : $err")
            continue
        end

        println("\n--- Test: L_type=$L_type  L=$(round(L_val,digits=4)) ---")

        # point de départ : MDS si dispo, sinon x0
        x_start = s_mds === nothing ? x0 : s_mds[2]

        forms = [
            #("new_standard", solve_new_standard),
            ("simple", solve_simple),
            ("simple_square", solve_simple_square),
            ("taylor", solve_taylor),
            ("taylor_square", solve_taylor_square),
            ("standard", solve_standard),
            #("polaire", solve_polaire),
            #("reformulee", solve_reformulee),
            #("surogate", solve_surogate),
            #("quadratique", solve_quadratique)
        ]

        for (name, fn) in forms
            try
                res = fn(inst.n, dim, inst.Q, c6, L_val, x_start)
                if res === nothing
                    println(@sprintf("[%s] no result", name))
                else
                    status, obj, x_sol = res
                    ry = erreur_rydberg_vraie(x_sol, inst.Q, c6, inst.n, dim)
                    println(@sprintf("[%s] status=%s  obj=%.6e  err_rydberg=%.6e", name, status, obj, ry))
                end
            catch err
                println("[$name] erreur: $(err)")
            end
        end
    end

    return nothing
end

function run_pasqal_tests()
    for folder in PASQAL_FOLDERS
        folder_path = joinpath(PASQAL_ROOT, folder)
        isdir(folder_path) || (println("Dossier absent: $folder_path"); continue)

        for inst_paths in pasqal_instance_paths(folder)
            run_one_pasqal_instance(inst_paths.txt_path, inst_paths.gml_path, folder)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pasqal_tests()
end
