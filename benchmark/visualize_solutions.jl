using JuMP, Ipopt, Random, LinearAlgebra, DataFrames, Statistics, Printf
using Plots, Plots.PlotMeasures

const c6 = 1
const dim = 2
const z_min = 0.00

include("utils.jl")
include("formulations.jl")

"""
    plot_configuration(x::AbstractMatrix, Q::AbstractMatrix, title::String="Configuration")

Visualize a 2D atomic configuration with edges representing connections.
"""
function plot_configuration(x::AbstractMatrix, Q::AbstractMatrix, title::String="Configuration")
    n = size(x, 1)
    p = plot(legend=false, aspect_ratio=:equal, title=title, 
             xaxis="x", yaxis="y", size=(600, 600),
             margin=5mm)
    
    # Draw edges (connections with non-zero weights)
    for (i, j) in active_pairs(Q)
        plot!(p, [x[i, 1], x[j, 1]], [x[i, 2], x[j, 2]], 
              color=:lightgray, alpha=0.5, label="")
    end
    
    # Draw nodes
    scatter!(p, x[:, 1], x[:, 2], markersize=8, color=:red, alpha=0.8, label="")
    
    # Add node labels
    for i in 1:n
        annotate!(p, x[i, 1], x[i, 2], text("$i", 8, :center))
    end
    
    return p
end

"""
    visualize_instance(txt_path::AbstractString, folder_name::AbstractString; 
                       formulations_to_plot=["simple", "taylor", "standard"])

Load a Pasqal instance and visualize solutions from multiple formulations.
"""
function visualize_instance(txt_path::AbstractString, folder_name::AbstractString; 
                           formulations_to_plot=["simple", "taylor", "standard"])
    
    inst = load_pasqal_instance(txt_path; gml_path=nothing, scale=100)

    x_init = generer_init(inst.n, dim, inst.L,seuil=0.0)
    
    n = inst.n
    L = inst.L
    Q = inst.Q
    println(L)
    
    println("\n════════════════════════════════════════════════════════════")
    println("Instance: $(basename(txt_path))")
    println("Dossier: $folder_name")
    println("n = $n, L = $L, nb_q_values = $(length(active_pairs(Q)))")
    println("════════════════════════════════════════════════════════════")
    
    # MDS reference
    plots_list = []
    s_mds = solve_mds(n, dim, Q, c6)
    if s_mds !== nothing
        _, x_mds = s_mds
        ry_mds = erreur_rydberg_vraie(x_mds, Q, c6, n, dim)
        p_mds = plot_configuration(x_mds, Q, 
                                   @sprintf("MDS (err=%.2e)", ry_mds))
        push!(plots_list, p_mds)
        println("[MDS] plotted")
    end
    
    # Test formulations
    L_fixe = 10^6
    L_adaptatif = 100 * sqrt(n)
    L_optimal = L_optimal_from_Q(Q, c6, n)

    
    L_values = [L_fixe, L_adaptatif, L_optimal]
    L_names = ["L_fixe", "L_adaptatif", "L_optimal"]
    
    for (L_val, L_name) in zip(L_values, L_names)
        println("\n--- $L_name = $(round(L_val, digits=2)) ---")
        
        for form_name in formulations_to_plot
            try
                res = nothing
                if form_name == "simple"
                    res = solve_simple(n, dim, Q, c6, L_val, x_init, z_min)
                elseif form_name == "simple_square"
                    res = solve_simple_square(n, dim, Q, c6, L_val, x_init, z_min)
                elseif form_name == "taylor"
                    res = solve_taylor(n, dim, Q, c6, L_val, x_init, z_min)
                elseif form_name == "taylor_square"
                    res = solve_taylor_square(n, dim, Q, c6, L_val, x_init, z_min)
                elseif form_name == "standard"
                    res = solve_standard(n, dim, Q, c6, L_val, x_init, z_min)
                # elseif form_name == "new_standard"
                #     res = solve_new_standard(n, dim, Q, c6, L_val, x_init, z_min)
                # elseif form_name == "polaire"
                #     res = solve_polaire(n, dim, Q, c6, L_val, x_init, z_min)
                # elseif form_name == "quadratique"
                #     res = solve_quadratique(n, dim, Q, c6, L_val, x_init, z_min)
                # elseif form_name == "reformulee"
                #     res = solve_reformulee(n, dim, Q, c6, L_val, x_init, z_min)
                # elseif form_name == "surogate"
                #     res = solve_surogate(n, dim, Q, c6, L_val, x_init,z_min)
                else
                    continue
                end

                res === nothing && continue
                status, obj, x_sol = res
                
                if x_sol !== nothing
                    err_ry = erreur_rydberg_vraie(x_sol, Q, c6, n, dim)
                    title = @sprintf("%s/%s (err=%.2e)", L_name, form_name, err_ry)
                    p = plot_configuration(x_sol, Q, title)
                    push!(plots_list, p)
                    println("  ✓ $form_name plotted (status=$status) obj=$(round(obj, digits=4)), err_ry=$(round(err_ry, digits=4))")
                else
                    println("  ✗ $form_name (status=$status)")
                end
            catch e
                println("  ✗ $form_name (error: $(typeof(e)))")
            end
        end
    end
    
    # Combine all plots
    n_plots = length(plots_list)
    n_cols = min(3, n_plots)
    n_rows = ceil(Int, n_plots / n_cols)
    
    p_combined = plot(plots_list..., layout=(n_rows, n_cols), size=(600*n_cols, 600*n_rows))
    
    # Save plot
    output_file = joinpath(@__DIR__, "visualizations", 
                          folder_name * "_" * splitext(basename(txt_path))[1] * ".png")
    mkpath(joinpath(@__DIR__, "visualizations"))
    savefig(p_combined, output_file)
    println("\n✓ Saved to: $output_file")
    
    return p_combined
end

function visualize_all_pasqal_instances(; formulations_to_plot=[
    "simple", "simple_square", "taylor", "taylor_square",
    "standard",
    # "new_standard", "polaire", "quadratique",
    # "reformulee", "surogate"
])
    pasqal_root = joinpath(@__DIR__, "pasqal")
    folders = filter(isdir, sort(readdir(pasqal_root; join=true)))

    for folder_path in folders
        folder_name = basename(folder_path)
        txt_files = sort(filter(f -> endswith(lowercase(f), ".txt"), readdir(folder_path; join=true)))

        println("\n==================== Folder: $folder_name ====================")
        for txt_path in txt_files
            try
                visualize_instance(txt_path, folder_name; formulations_to_plot=formulations_to_plot)
            catch e
                println("✗ $txt_path (error: $(typeof(e)))")
            end
        end
    end
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    visualize_all_pasqal_instances()
end
