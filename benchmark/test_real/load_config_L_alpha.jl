# ═══════════════════════════════════════════════════════════════
#  Charge config_L_alpha.toml (+ surcharge ENV si présente)
#
#  Usage :
#    include(joinpath(@__DIR__, "load_config_L_alpha.jl"))
#    CFG = load_L_alpha_config(; profile=:full)   # :full | :xr | :simple
# ═══════════════════════════════════════════════════════════════

using TOML

function _cfg_get(d::AbstractDict, key::AbstractString, default)
    return haskey(d, key) ? d[key] : default
end

function _cfg_abspath(p::AbstractString, base::AbstractString)
    isempty(p) && return base
    return isabspath(p) ? p : normpath(joinpath(base, p))
end

function _cfg_floats(v)
    if v isa AbstractVector
        return Float64[Float64(x) for x in v]
    end
    return parse.(Float64, split(string(v), ","; keepempty=false))
end

function _cfg_ints(v)
    if v isa AbstractVector
        return Int[Int(x) for x in v]
    end
    return parse.(Int, split(string(v), ","; keepempty=false))
end

function _cfg_bool(v)
    v isa Bool && return v
    s = lowercase(strip(string(v)))
    return s in ("1", "true", "yes", "on")
end

function _cfg_forms(v)
    if v isa AbstractVector
        return join(string.(v), ",")
    end
    return string(v)
end

"""
    load_L_alpha_config(; profile=:full, path=...)

Lit `config_L_alpha.toml`. Les variables d'environnement (N_VALUES, ALPHAS,
RUN_EXP1, OUT_DIR, …) priment sur le fichier — pratique pour Joyeux.
"""
function load_L_alpha_config(; profile::Symbol=:full,
                             path::AbstractString=joinpath(@__DIR__, "config_L_alpha.toml"))
    isfile(path) || error("Config introuvable : $path")
    raw = TOML.parsefile(path)
    phys = _cfg_get(raw, "physics", Dict{String,Any}())
    common = _cfg_get(raw, "common", Dict{String,Any}())
    sect = _cfg_get(raw, string(profile), Dict{String,Any}())
    base = @__DIR__

    # ── helpers ENV > TOML ──
    env_str(k, d) = haskey(ENV, k) ? ENV[k] : string(d)
    env_int(k, d) = haskey(ENV, k) ? parse(Int, ENV[k]) : Int(d)
    env_float(k, d) = haskey(ENV, k) ? parse(Float64, ENV[k]) : Float64(d)
    env_bool(k, d) = haskey(ENV, k) ? (ENV[k] == "1" || lowercase(ENV[k]) == "true") : _cfg_bool(d)
    env_ints(k, d) = haskey(ENV, k) ? _cfg_ints(ENV[k]) : _cfg_ints(d)
    env_floats(k, d) = haskey(ENV, k) ? _cfg_floats(ENV[k]) : _cfg_floats(d)

    n_values = env_ints("N_VALUES", _cfg_get(sect, "n_values", [4, 10]))
    n_instances = env_int("N_INSTANCES", _cfg_get(common, "n_instances", 5))
    alphas = env_floats("ALPHAS", _cfg_get(sect, "alphas", [1e-6, 1e-4, 1e-2, 1.0]))
    l_init = env_float("L_INIT", _cfg_get(common, "l_init", 10.0))
    l_chosen = env_float("L_CHOSEN", _cfg_get(common, "l_chosen", 10.0))
    forms = haskey(ENV, "FORMS") ? ENV["FORMS"] :
            _cfg_forms(_cfg_get(sect, "forms", _cfg_get(common, "forms",
                "Standard,Simple,Simple_square,Taylor,Taylor_square")))
    zmin_match_tol = env_float("ZMIN_MATCH_TOL", _cfg_get(common, "zmin_match_tol", 0.2))
    seed = env_int("SEED", _cfg_get(common, "seed", 42))

    run_exp1 = env_bool("RUN_EXP1", _cfg_get(sect, "run_exp1", true))
    run_exp2 = env_bool("RUN_EXP2", _cfg_get(sect, "run_exp2", true))
    run_exp3 = env_bool("RUN_EXP3", _cfg_get(sect, "run_exp3", false))

    out_dir_raw = haskey(ENV, "OUT_DIR") ? ENV["OUT_DIR"] :
                  string(_cfg_get(sect, "out_dir", base))
    out_dir = _cfg_abspath(out_dir_raw, base)

    pasqal_default = joinpath(base, "..", "pasqal")
    pasqal_raw = haskey(ENV, "PASQAL_ROOT") ? ENV["PASQAL_ROOT"] :
                 string(_cfg_get(sect, "pasqal_root",
                        _cfg_get(common, "pasqal_root", pasqal_default)))
    pasqal_root = _cfg_abspath(pasqal_raw, base)

    out_csv_name = env_str("OUT_CSV_NAME",
        _cfg_get(sect, "out_csv_name", "exp_L_alpha_full_results.csv"))
    out_log_name = env_str("OUT_LOG_NAME",
        _cfg_get(sect, "out_log_name", "exp_L_alpha_full.log"))
    out_runs_name = env_str("OUT_RUNS_NAME",
        _cfg_get(sect, "out_runs_name", "exp_L_alpha_xr_runs.csv"))
    out_pairs_name = env_str("OUT_PAIRS_NAME",
        _cfg_get(sect, "out_pairs_name", "exp_L_alpha_xr_pairs.csv"))

    return (
        profile = profile,
        config_path = path,
        c6 = Float64(_cfg_get(phys, "c6", 1.0)),
        dim = Int(_cfg_get(phys, "dim", 2)),
        z_min = Float64(_cfg_get(phys, "z_min", 0.04)),
        d_min = Float64(_cfg_get(phys, "d_min", 0.04)),
        n_values = n_values,
        n_instances = n_instances,
        alphas = alphas,
        l_init = l_init,
        l_chosen = l_chosen,
        forms = forms,
        zmin_match_tol = zmin_match_tol,
        seed = seed,
        run_exp1 = run_exp1,
        run_exp2 = run_exp2,
        run_exp3 = run_exp3,
        out_dir = out_dir,
        pasqal_root = pasqal_root,
        out_csv_name = String(out_csv_name),
        out_log_name = String(out_log_name),
        out_runs_name = String(out_runs_name),
        out_pairs_name = String(out_pairs_name),
        out_csv = joinpath(out_dir, String(out_csv_name)),
        out_log = joinpath(out_dir, String(out_log_name)),
        out_runs = joinpath(out_dir, String(out_runs_name)),
        out_pairs = joinpath(out_dir, String(out_pairs_name)),
    )
end
