function generer_init(n, dim, L; seuil=z_min)
    x = zeros(n, dim)
    for i in 1:n
        x[i, :] = L * rand(dim)
        # On teste le point i contre tous les points précédents (1 à i-1)
        j = 1
        while j < i
            if norm(x[i, :] - x[j, :]) < seuil
                # Collision détectée : on régénère le point i
                x[i, :] = L * rand(dim)
                # On redémarre la vérification pour i à partir de j=1
                j = 1
            else
                j += 1
            end
        end
    end
    return x
end

function erreur_rydberg_vraie(x_sol, Q, c6, n, dim)

    try
        val = 0.0
        for i in 1:n, j in i+1:n
            d = norm(x_sol[i,:] - x_sol[j,:])
            val += (c6 / d^6 - Q[i,j])^2
        end
        return val
    catch
        return Inf
    end
end

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

function rmse_dist_safe(x, x_ref, n)
    x === nothing && return Inf
    try
        return rmse_dist(x, x_ref, n)
    catch
        return Inf
    end
end

function matrix_to_csv_str(M::AbstractMatrix)
    rows = String[]
    for i in 1:size(M, 1)
        push!(rows, join(string.(M[i, :]), ","))
    end
    return join(rows, ";")
end

function csv_str_to_matrix(s::AbstractString)
    isempty(s) && return nothing
    rows = split(s, ";")
    data = [parse.(Float64, split(row, ",")) for row in rows]
    return reduce(vcat, [reshape(r, 1, :) for r in data])
end


# ═══════════════════════════════════════════════════════════════
#  §2a  PASQAL — chargement des instances
#  - .txt : matrice Q sous forme d'arêtes pondérées
#  - .gml : positions initiales (utile comme x_init)
# ═══════════════════════════════════════════════════════════════
function read_pasqal_txt_q(txt_path::AbstractString; scale::Real = 1)
    open(txt_path, "r") do io
        header = split(strip(readline(io)))
        length(header) >= 1 || error("Fichier Pasqal invalide: $txt_path")
        n = parse(Int, header[1])
        Q = zeros(Float64, n, n)

        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            parts = split(s)
            length(parts) < 3 && continue
                i = parse(Int, parts[1]) + 1
                j = parse(Int, parts[2]) + 1
                raww = parse(Float64, parts[3])
                # Ignorer les diagonales et poids non-positifs (entrées linéaires ou métadonnées)
                if i == j || raww <= 0
                    continue
                end
                w = raww / scale
                Q[i, j] = w
                Q[j, i] = w
        end

        return Q
    end
end

function read_pasqal_gml_positions(gml_path::AbstractString)
    ids = Int[]
    xs  = Float64[]
    ys  = Float64[]

    current_id = nothing
    current_x  = nothing
    current_y  = nothing

    open(gml_path, "r") do io
        for line in eachline(io)
            s = strip(line)
            if startswith(s, "id ")
                current_id = parse(Int, split(s)[2])
            elseif startswith(s, "pos ")
                v = parse(Float64, split(s)[2])
                if current_x === nothing
                    current_x = v
                else
                    current_y = v
                end
            elseif s == "]"
                if current_id !== nothing && current_x !== nothing && current_y !== nothing
                    push!(ids, current_id)
                    push!(xs, current_x)
                    push!(ys, current_y)
                end
                current_id = nothing
                current_x = nothing
                current_y = nothing
            end
        end
    end

    length(ids) == 0 && error("Aucune position trouvée dans $gml_path")
    n = maximum(ids) + 1
    X = zeros(Float64, n, 2)
    for k in eachindex(ids)
        X[ids[k] + 1, 1] = xs[k]
        X[ids[k] + 1, 2] = ys[k]
    end
    return X
end

function load_pasqal_instance(txt_path::AbstractString; gml_path::Union{Nothing, AbstractString} = nothing,
                              scale::Real = 1000.0)
    Q = read_pasqal_txt_q(txt_path; scale=scale)

    x_ref = gml_path === nothing ? nothing : read_pasqal_gml_positions(gml_path)

    n = size(Q, 1)
    L = x_ref === nothing ? L_optimal_from_Q(Q, 1.0, n) : maximum(abs.(x_ref)) + 1.0
    return (Q=Q, x_ref=x_ref, n=n, L=L, txt_path=txt_path, gml_path=gml_path)
end

function load_pasqal_folder(folder::AbstractString; scale::Real = 1000.0)
    txt_files = filter(f -> endswith(lowercase(f), ".txt"), readdir(folder; join=true))
    isempty(txt_files) && error("Aucun fichier .txt trouvé dans $folder")

    instances = NamedTuple[]
    for txt_path in sort(txt_files)
        base = splitext(txt_path)[1]
        gml_path = isfile(base * ".gml") ? base * ".gml" : (isfile(joinpath(folder, "graph.gml")) ? joinpath(folder, "graph.gml") : nothing)
        push!(instances, load_pasqal_instance(txt_path; gml_path=gml_path, scale=scale))
    end
    return instances
end

function list_pasqal_instances(root::AbstractString)
    sort(filter(isdir, readdir(root; join=true)))
end

function active_pairs(Q; tol::Real = 0.0)
    n = size(Q, 1)
    [(i, j) for i in 1:n for j in i+1:n if Q[i, j] > tol]
end


# ═══════════════════════════════════════════════════════════════
#  §2b  BRUIT MULTIPLICATIF
#  Seul ajout au code original : fonction pour bruiter Q
# ═══════════════════════════════════════════════════════════════
function ajouter_bruit(Q, theta_bruit, n)
    Q_bruit = copy(Q)
    for i in 1:n, j in i+1:n
        Q_bruit[i,j] = max(Q[i,j] * (1 + theta_bruit * randn()))
        Q_bruit[j,i] = Q_bruit[i,j]
    end
    return Q_bruit
end

function Q_bruit_dist(n, c6, L)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        d = rand() * L 
        Q[i,j] = Q[j,i] = c6 / d^6
    end
    return Q
end

function build_Q_matrix(x, c6, n)
    Q = zeros(n, n)
    for i in 1:n, j in i+1:n
        d = norm(x[i,:] - x[j,:])
        Q[i,j] = Q[j,i] = c6 / d^6
    end
    return Q
end

function L_optimal_from_Q(Q, c6, n)
    vals = [Q[i,j] for i in 1:n for j in i+1:n if Q[i,j] > 0]
    isempty(vals) && error("Q ne contient aucune interaction positive")
    qmin = minimum(vals)
    return ((c6 / qmin)^(1/6))
end

"""Distance minimale optimale : zmin = (c6 / Q_max)^{1/6}."""
function zmin_optimal(Q, c6)
    n = size(Q, 1)
    vals = [Q[i, j] for i in 1:n for j in i+1:n if Q[i, j] > 0]
    isempty(vals) && error("Q ne contient aucune interaction positive")
    return (c6 / maximum(vals))^(1 / 6)
end

function min_pairwise_distance(x)
    n = size(x, 1)
    dmin = Inf
    for i in 1:n, j in i+1:n
        d = norm(x[i, :] - x[j, :])
        d < dmin && (dmin = d)
    end
    return dmin
end

function L_value(L_type::AbstractString, n::Int; Q=nothing, c6=1.0)
    if L_type == "L_fixe"
        return 10.0
    elseif L_type == "L_adaptatif"
        return 100.0 * sqrt(n)
    elseif L_type == "L_optimal"
        Q === nothing && error("L_optimal nécessite la matrice Q")
        return L_optimal_from_Q(Q, c6, size(Q, 1))
    else
        error("Type de L inconnu : $L_type")
    end
end

function normalize_formulation_name(s::AbstractString)
    t = lowercase(strip(s))
    t in ("std", "standard") && return "Standard"
    # t in ("new_std", "new_standard") && return "new_standard"
    # t in ("reform", "reformulee", "reformulée") && return "Reformulee"
    # t in ("pol", "polaire") && return "Polaire"
    # t in ("surogate", "surrogate") && return "Surogate"
    # t in ("quad", "quadratique") && return "Quadratique"
    t in ("taylor",) && return "Taylor"
    t in ("taylor_square", "taylor_sq") && return "Taylor_square"
    t in ("simple_square", "simple_sq") && return "Simple_square"
    t in ("simple",) && return "Simple"
    t in ("mds", "mds_exact", "mdsexact") && return "MDS_Exact"
    return s
end

display_formulation_name(form::AbstractString) = form

safe(x) = isinf(x) || isnan(x) ? NaN : x

finite_mean(v) = begin
    xs = filter(isfinite, v)
    isempty(xs) ? NaN : mean(xs)
end

function aggregate_benchmark(df::DataFrame)
    combine(
        groupby(df, [:n, :L_type, :strategy, :formulation]),
        :L_val       => finite_mean => :L_val,
        :obj_solver  => finite_mean => :obj_solver,
        :obj_rydberg => finite_mean => :obj_rydberg,
        :rmse_dist   => finite_mean => :rmse_dist,
        :n_restarts  => finite_mean => :n_restarts,
        :time_s      => finite_mean => :time_s,
        :status      => (s -> join(unique(s), ", ")) => :status,
    )
end

function print_benchmark_means(df_mean::DataFrame, n_instances::Int)
    println("\n" * "═"^70)
    println("  MOYENNES SUR $n_instances INSTANCES — obj_rydberg et temps moyen")
    println("═"^70)
    for grp in groupby(df_mean, [:n, :L_type])
        r = grp[1, :]
        @printf("\n  n=%-3d  %s\n", r.n, r.L_type)
        min_ry = minimum(replace(grp.obj_rydberg, Inf => 1e300))
        for row in eachrow(grp)
            ry_cmp = isinf(row.obj_rydberg) ? 1e300 : row.obj_rydberg
            marker = ry_cmp == min_ry ? " ← ★" : ""
            @printf("    %-12s [%-11s]  obj_ry=%+.4e  rmse=%.4e  t_moy=%7.4fs%s\n",
                    display_formulation_name(row.formulation), row.strategy,
                    row.obj_rydberg, row.rmse_dist, row.time_s, marker)
        end
    end
    println("═"^70)
end

function run_benchmark(n_values, dim, c6; formulations::Vector{String}=["Standard", "Taylor", "Taylor_square", "Simple_square", "Simple", "MDS_Exact"], L_types::Vector{String}=ALL_L_TYPES, n_instances::Int=10)
    df = DataFrame(
        instance     = Int[],
        n            = Int[],
        L_type       = String[],
        L_val        = Float64[],
        strategy     = String[],
        formulation  = String[],
        obj_solver   = Float64[],
        obj_rydberg  = Float64[],
        rmse_dist    = Float64[],
        n_restarts   = Int[],
        time_s       = Float64[],
        status       = String[]
    )

    formulations = normalize_formulation_name.(formulations)
    forms_set = Set(formulations)
    nlp_forms = Pair{String, Any}[]
    "Standard" in forms_set    && push!(nlp_forms, "Standard"    => solve_standard)
    # "new_standard" in forms_set && push!(nlp_forms, "new_standard" => solve_new_standard)
    # "Reformulee" in forms_set  && push!(nlp_forms, "Reformulee"  => solve_reformulee)
    # "Polaire" in forms_set     && push!(nlp_forms, "Polaire"     => solve_polaire)
    # "Surogate" in forms_set    && push!(nlp_forms, "Surogate"    => solve_surogate)
    # "Quadratique" in forms_set && push!(nlp_forms, "Quadratique" => solve_quadratique)
    "Taylor" in forms_set        && push!(nlp_forms, "Taylor"        => solve_taylor)
    "Taylor_square" in forms_set && push!(nlp_forms, "Taylor_square" => solve_taylor_square)
    "Simple_square" in forms_set && push!(nlp_forms, "Simple_square" => solve_simple_square)
    "Simple" in forms_set        && push!(nlp_forms, "Simple"        => solve_simple)
    include_mds = "MDS_Exact" in forms_set



for inst in 1:n_instances
    println("\n── Instance $inst / $n_instances ──")
    for n in n_values
        for L_type in L_types
            if L_type == "L_optimal"
                x_vrais = generer_init(n, dim, L_SEED)
                Q_local = build_Q_matrix(x_vrais, c6, n)
                L = L_optimal_from_Q(Q_local, c6, n)
            else
                L = L_value(L_type, n; c6=c6)
                x_vrais = generer_init(n, dim, L)
                Q_local = build_Q_matrix(x_vrais, c6, n)
            end

            println("\n" * "═"^62)
            @printf("  n=%-3d  |  %s  (L=%.4f)\n", n, L_type, L)
            println("═"^62)

            n_rs           = nb_restarts_bench(n)
            x_inits_shared = [generer_init(n, dim, L) for _ in 1:n_rs]

            x_mds = nothing
            if include_mds
                # ── MDS : un seul appel algébrique ─────────────────
                t_mds = @elapsed begin
                    _, x_mds = solve_mds(n, dim, Q_local, c6)
                end
                ry_mds = erreur_rydberg_vraie(x_mds, Q_local, c6, n, dim)
                rmse_mds = rmse_dist_safe(x_mds, x_vrais, n)
                @printf("    [%-12s] obj_rydberg=%+.3e  rmse_dist=%.4e  t=%8.5fs  [EXACT]\n",
                        "MDS_Exact", ry_mds, rmse_mds, t_mds)
                push!(df, (inst, n, L_type, L, "exact", "MDS_Exact",
                           0.0, ry_mds, rmse_mds, 1, t_mds, "MDS_EXACT"))
            end

            # ── NLP : single et multistart ──────────────────────
            for (strategy_name, use_multistart) in [("single",     false),
                                                     ("multistart", true)]
                println("\n  ── Stratégie : $strategy_name ──")
                sols_config = Dict{String, Union{Nothing, Matrix{Float64}}}()

                for (form_name, solver_fn) in nlp_forms
                    obj_s, obj_ry, stat_s, x_sol, n_used, t_s =run_formulation(solver_fn, n, dim, Q_local, c6, L, x_inits_shared, use_multistart)
                    sols_config[form_name] = x_sol
                    rmse_s = rmse_dist_safe(x_sol, x_vrais, n)
                    @printf("    [%-12s] obj_solver=%+.3e  obj_rydberg=%+.3e  rmse_dist=%.4e  t=%6.2fs  [%s]\n",
                            display_formulation_name(form_name), obj_s, obj_ry, rmse_s, t_s,
                            stat_s[1:min(20, length(stat_s))])
                    push!(df, (inst, n, L_type, L, strategy_name, form_name,
                               obj_s, obj_ry, rmse_s, n_used, t_s, stat_s))
                end

                if include_mds
                    println("\n    ┌── Injection croisée ──────────────────────────┐")
                    scores = Float64[]
                    all_forms = vcat(nlp_forms, [("MDS_Exact", nothing)])
                    sols_config["MDS_Exact"] = x_mds
                    for (f, _) in all_forms
                        ry = erreur_rydberg_vraie(sols_config[f], Q_local, c6, n, dim)
                        rmse_f = rmse_dist_safe(sols_config[f], x_vrais, n)
                        push!(scores, ry)
                        @printf("    │  x_sol[%-12s] → err_Rydberg = %+.3e  rmse_dist = %.4e\n", display_formulation_name(f), ry, rmse_f)
                    end
                    best_form = all_forms[argmin(scores)][1]
                    println("    │  → GAGNANT : $(display_formulation_name(best_form))")
                    println("    └──────────────────────────────────────────────────┘")
                end
            end
        end
    end
    end

    df_mean = aggregate_benchmark(df)
    print_benchmark_means(df_mean, n_instances)
    return df_mean, df
end

function positive_finite_values(series_list)
    vals = Float64[]
    for series in series_list
        append!(vals, filter(x -> isfinite(x) && x > 0, series))
    end
    return vals
end

function _metric_rows(df, n, L_type, strategy, form, col::Symbol)
    rows = filter(r -> r.n == n && r.L_type == L_type &&
                       r.strategy == strategy && r.formulation == form,
                  eachrow(df))
    isempty(rows) ? NaN : safe(finite_mean(getproperty.(rows, col)))
end

get_ry(df, n, L_type, strategy, form)    = _metric_rows(df, n, L_type, strategy, form, :obj_rydberg)
get_time(df, n, L_type, strategy, form)  = _metric_rows(df, n, L_type, strategy, form, :time_s)
get_rmse(df, n, L_type, strategy, form)  = _metric_rows(df, n, L_type, strategy, form, :rmse_dist)

function plot_rmse_multistart(df, L_type, L_label)
    series = [
        [get_rmse(df, n, L_type, "multistart", form) for n in n_values]
        for form in FORMS_NLP
    ]
    vals_mds = [get_rmse(df, n, L_type, "exact", "MDS_Exact") for n in n_values]
    all_vals = positive_finite_values(vcat(series, [vals_mds]))

    p = if !isempty(all_vals)
        plot(title="RMSE distances — Multistart + MDS, $L_label",
             xlabel="n (atomes)", ylabel="RMSE distances (log₁₀)",
             legend=:topleft, yscale=:log10,
             ylims=(minimum(all_vals)/10, maximum(all_vals)*10),
             xticks=(1:length(n_values), n_str),
             yminorgrid=false, size=(600, 400))
    else
        plot(title="RMSE distances — Multistart + MDS, $L_label",
             xlabel="n (atomes)", ylabel="RMSE distances",
             legend=:topleft,
             xticks=(1:length(n_values), n_str),
             yminorgrid=false, size=(600, 400))
    end

    for (form, lab) in zip(FORMS_NLP, LABELS_NLP)
        vals = get_rmse.(Ref(df), n_values, Ref(L_type), Ref("multistart"), Ref(form))
        plot!(p, 1:length(n_values), vals,
              label=lab, color=COULEURS[form],
              marker=:circle, markersize=6, linewidth=2,
              linestyle=LINE_STYLES_NLP[findfirst(==(form), FORMS_NLP)])
    end
    plot!(p, 1:length(n_values), vals_mds,
          label="MDS Exact", color=COULEURS["MDS_Exact"],
          marker=:star5, markersize=9, linewidth=2, linestyle=:dash)
    return p
end

function aggregate_bruit_benchmark(df::DataFrame)
    combine(
        groupby(df, [:n, :L_type, :sigma, :formulation]),
        :L_val        => finite_mean => :L_val,
        :err_vs_vrai  => finite_mean => :err_vs_vrai,
        :err_vs_bruit => finite_mean => :err_vs_bruit,
        :rmse_dist    => finite_mean => :rmse_dist,
        :time_s       => finite_mean => :time_s,
    )
end

function print_bruit_means(df_mean::DataFrame, n_instances::Int)
    println("\n\n" * "═"^70)
    println("  Q_bruit_dist — moyennes sur $n_instances instances")
    println("  err_vs_bruit (ajustement à Q aléatoire) · temps moyen")
    println("═"^70)
    for grp in groupby(df_mean, [:n, :L_type, :sigma])
        r = grp[1, :]
        @printf("\n  n=%-3d  %s  theta_bruit=%.0f%%\n", r.n, r.L_type, r.sigma * 100)
        min_ry = minimum(replace(grp.err_vs_bruit, Inf => 1e300))
        for row in eachrow(grp)
            ry_cmp = isinf(row.err_vs_bruit) ? 1e300 : row.err_vs_bruit
            marker = ry_cmp == min_ry ? " ← ★ MEILLEUR" : ""
            @printf("    %-12s  err_bruit=%+.4e  t_moy=%7.4fs%s\n",
                    display_formulation_name(row.formulation), row.err_vs_bruit,
                    row.time_s, marker)
        end
    end
    println("═"^70)
end

function get_bruit_metric(df, n, L_type, sigma, form, metric::Symbol)
    rows = filter(r -> r.n == n && r.L_type == L_type &&
                       r.sigma == sigma && r.formulation == form,
                  eachrow(df))
    isempty(rows) ? NaN : safe(finite_mean(getproperty.(rows, metric)))
end

function plot_bruit_grid(df, metric::Symbol, metric_label::String)
    subplots = Any[]
    for n in n_values
        for L_type in ALL_L_TYPES
            series_list = [
                [get_bruit_metric(df, n, L_type, sigma, form, metric) for sigma in bruit_niveaux]
                for form in FORMS_ALL
            ]
            vals_pos = positive_finite_values(series_list)

            p = if !isempty(vals_pos)
                plot(title="n=$n — $L_type — $metric_label",
                     xlabel="σ bruit", ylabel=metric_label,
                     yscale=:log10,
                     ylims=(minimum(vals_pos) / 10, maximum(vals_pos) * 10),
                     xticks=(bruit_niveaux, string.(bruit_niveaux)),
                     legend=:topright, yminorgrid=false, size=(600, 380))
            else
                plot(title="n=$n — $L_type — $metric_label",
                     xlabel="σ bruit", ylabel=metric_label,
                     xticks=(bruit_niveaux, string.(bruit_niveaux)),
                     legend=:topright, yminorgrid=false, size=(600, 380))
            end

            for (form, lab) in zip(FORMS_ALL, LABELS_ALL)
                vals = [get_bruit_metric(df, n, L_type, sigma, form, metric) for sigma in bruit_niveaux]
                plot!(p, bruit_niveaux, vals,
                      label=lab, color=COULEURS[form],
                      marker=(form == "MDS_Exact" ? :star5 : :circle),
                      markersize=(form == "MDS_Exact" ? 8 : 6),
                      linewidth=2,
                      linestyle=(form == "MDS_Exact" ? :dash : LINE_STYLES_NLP[findfirst(==(form), FORMS_NLP)]))
            end

            push!(subplots, p)
        end
    end

    n_sub = length(subplots)
    n_cols = (n_sub % 2 == 0) ? 2 : 1
    n_rows = cld(n_sub, n_cols)

    return plot(subplots..., layout=(n_rows, n_cols),
                size=(1300, 500 * n_rows),
                margin=6Plots.mm,
                plot_title="Benchmark bruit — $metric_label")
end


function align_to_reference(X_ref::Matrix{Float64}, X_est::Matrix{Float64})
    μ_ref = vec(mean(X_ref, dims=1))
    μ_est = vec(mean(X_est, dims=1))

    Xr = X_ref .- reshape(μ_ref, 1, :)
    Xe = X_est .- reshape(μ_est, 1, :)

    M = transpose(Xe) * Xr
    F = svd(M)
    R = F.U * F.Vt

    if det(R) < 0
        S = Matrix{Float64}(I, size(R, 1), size(R, 2))
        S[end, end] = -1.0
        R = F.U * S * F.Vt
    end

    Xa = Xe * R
    Xa .+= reshape(μ_ref, 1, :)
    return Xa
end

function plot_mds_positions_grid(positions::Dict, sigma::Float64)
    subplots = Any[]
    for n in n_values
        for L_type in ALL_L_TYPES
            key = (n, L_type, sigma)
            if !haskey(positions, key)
                continue
            end
            x_vrai, x_mds = positions[key]
            x_mds_aligne = align_to_reference(x_vrai, x_mds)

            p = scatter(x_vrai[:, 1], x_vrai[:, 2],
                        label="X vrai",
                        color=:black,
                        marker=:diamond,
                        markersize=7,
                        markerstrokewidth=0,
                        xlabel="x", ylabel="y",
                        title="n=$n — $L_type",
                        legend=:topright,
                        aspect_ratio=:equal,
                        size=(520, 420))

            scatter!(p, x_mds_aligne[:, 1], x_mds_aligne[:, 2],
                     label="MDS Exact (x_opt)",
                     color=COULEURS["MDS_Exact"],
                     marker=:star5,
                     markersize=8,
                     markerstrokewidth=0)

            for i in 1:n
                annotate!(p, x_vrai[i, 1], x_vrai[i, 2], text(string(i), 7, :black))
            end

            push!(subplots, p)
        end
    end
end