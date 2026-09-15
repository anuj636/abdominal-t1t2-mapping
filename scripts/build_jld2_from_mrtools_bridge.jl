using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NPZ
using JLD2
using ReconBMRR

const BRIDGE_DEBUG_LABELS = get(ENV, "BRIDGE_DEBUG_LABELS", "0") == "1"
const BRIDGE_ENFORCE_EXPECTED_VIEWS = get(ENV, "BRIDGE_ENFORCE_EXPECTED_VIEWS", "0") == "1"
const BRIDGE_EXPECTED_VIEW_COUNT = try
    parse(Int, get(ENV, "BRIDGE_EXPECTED_VIEW_COUNT", "28"))
catch
    28
end
const BRIDGE_INTERLEAVE_SOURCE = lowercase(get(ENV, "BRIDGE_INTERLEAVE_SOURCE", "auto"))

function build_label_lookup_table(n::Int)
    return Any[
        reshape(Float64.(collect(1:n)), 1, :),  # accImagData indices
        reshape(Float64[], 1, 0),               # rejImagData indices
        reshape(Float64[], 1, 0),               # phaseCorrData indices
        reshape(Float64[], 1, 0),               # freqCorrData indices
        reshape(Float64[], 1, 0),               # noiseData indices
    ]
end

function build_scan_parameters_base()
    return Dict{Symbol,Any}(
        :AcqMode => "Cartesian",
        :FieldStrength => Float32(3.0),
        :TR => Float32(0.0),
        :FlipAngle => Float32(0.0),
        :TE => Float32[],
        :AcqVoxelSize => Float32[1.0, 1.0, 1.0],
        :RecVoxelSize => Float32[1.0, 1.0, 1.0],
        :FOV => Float32[1.0, 1.0, 1.0],
        :KxRange => Int32[],
        :KyRange => Int32[],
        :KzRange => Int32[],
        :encodingSize => Int32[1, 1, 1],
        :TFEfactor => 1,
    )
end

function build_recon_parameters_base(scan::Dict{Symbol,Any})
    return Dict{Symbol,Any}(
        :cuda => false,
        :cudaSolver => false,
        :useDoublePrecision => false,
        :artificalUndersampling => 0,
        :export => Dict{Symbol,Any}(),
        :iterativeReconParams => ReconBMRR.setIterativeReconParams(),
        :motionGating => false,
        :motionGatingParams => Dict{Symbol,Any}(),
        :motionStatesRecon => nothing,
        :noisePreWhitening => false,
        :upsampleRecVoxelSize => scan[:AcqMode] == "Radial" ? true : nothing,
        :removeOversampling => true,
        :kspOrdered => false,
        :coilSensitivities => false,
    )
end

function deep_merge(base::AbstractDict, override::AbstractDict; include_new_keys::Bool=true)
    merged = Dict{Symbol,Any}()
    for (k, v) in base
        merged[k] = deepcopy(v)
    end
    for (k, v_override) in override
        if haskey(merged, k) && merged[k] isa AbstractDict && v_override isa AbstractDict
            merged[k] = deep_merge(merged[k], v_override; include_new_keys=true)
        elseif haskey(merged, k) || include_new_keys
            merged[k] = deepcopy(v_override)
        end
    end
    return merged
end

function build_kdatapreprocessed_from_bridge(
    acc_imag_data::Matrix{Complex{Float32}},
    ky::Vector{Int32},
    kz::Vector{Int32},
    echo::Vector,
    dyn::Vector,
    chan::Vector,
    extr1::Vector,
    num_kx::Int;
    nominal_ky_min::Union{Int,Nothing}=nothing,
    nominal_ky_max::Union{Int,Nothing}=nothing,
    nominal_kz_min::Union{Int,Nothing}=nothing,
    nominal_kz_max::Union{Int,Nothing}=nothing,
)
    """
    Build KdataPreprocessed directly from MRTOOLS bridge data, bypassing sortData().
    Returns tuple: (kdata_obj, trajectory)
    """
    
    # Get unique values and ranges. Reviewer-flagged issue (see TODO.md
    # 2026-09-14 "200x75x141 k-space dimensions"): the ACQUIRED extent (this
    # scan's accelerated CASPR trajectory never samples the outermost shell)
    # is smaller than the NOMINAL encoding matrix from the .sin header. Using
    # the nominal bounds (when available) keeps the reconstruction grid at
    # the sequence's true designed FOV; the 2 missing outer shells on each
    # side simply stay exact zero (already how undersampled positions are
    # handled everywhere else in this pipeline).
    ky_min = nominal_ky_min !== nothing ? nominal_ky_min : minimum(ky)
    ky_max = nominal_ky_max !== nothing ? nominal_ky_max : maximum(ky)
    kz_min = nominal_kz_min !== nothing ? nominal_kz_min : minimum(kz)
    kz_max = nominal_kz_max !== nothing ? nominal_kz_max : maximum(kz)
    
    num_ky = ky_max - ky_min + 1
    num_kz = kz_max - kz_min + 1
    num_echoes = maximum(echo) + 1
    num_dyn = maximum(dyn) + 1
    num_chan = maximum(chan) + 1
    num_inter = maximum(extr1) + 1
    
    # Initialize 7D array: (kx, ky, kz, echoes, dynamics, channels, interleaves)
    kdata = zeros(ComplexF32, num_kx, num_ky, num_kz, num_echoes, num_dyn, num_chan, num_inter)
    
    # Initialize trajectory: (2, ky, kz, echoes, dynamics, interleaves)
    # profileOrder[1, :] = ky indices, profileOrder[2, :] = kz indices
    profileOrder = zeros(Int, 2, num_ky, num_kz, num_echoes, num_dyn, num_inter)
    
    # Map profiles into 7D array and build trajectory
    for i = 1:size(acc_imag_data, 2)
        ky_idx = ky[i] - ky_min + 1
        kz_idx = kz[i] - kz_min + 1
        echo_idx = echo[i] + 1
        dyn_idx = dyn[i] + 1
        chan_idx = chan[i] + 1
        inter_idx = extr1[i] + 1
        
        # Place profile at (ky_idx, kz_idx) for all other dimensions
        kdata[:, ky_idx, kz_idx, echo_idx, dyn_idx, chan_idx, inter_idx] .= acc_imag_data[:, i]
        
        # Record profile coordinates (convert back to original ky/kz values for reference)
        profileOrder[1, ky_idx, kz_idx, echo_idx, dyn_idx, inter_idx] = ky[i]
        profileOrder[2, ky_idx, kz_idx, echo_idx, dyn_idx, inter_idx] = kz[i]
    end
    
    return ReconBMRR.KdataPreprocessed(kdata), profileOrder
end

function build_kdatapreprocessed_caspr_subspace_from_bridge(
    acc_imag_data::Matrix{Complex{Float32}},
    ky::Vector{Int32},
    kz::Vector{Int32},
    echo::Vector,
    tfe_slot::Vector,
    shot_repeat::Vector,
    contr::Vector,
    chan::Vector,
    num_kx::Int,
)
    """
    Build KdataPreprocessed indexed by CASPR shot/TFE structure instead of a
    dense (ky,kz) Cartesian grid. This is required for CuCasprSubspaceOp,
    which assumes profiles[2, numTFE, numShots, numEchoesContr] with:
      - numTFE   = readout position within a shot (dictionary's TFE axis)
      - numShots = repeat/execution index of a given (dyn,card) shot
      - contr    = 0-based dyn*num_card+card, matching the dictionary's
                   (TFE fastest, phase, delay slowest) flatten convention
    See TODO.md "Track B" sections for the empirical derivation of
    tfe_slot/shot_repeat from raw seq_nr, and Preprocessing.jl's
    subspaceBasis! for the confirmed dictionary flatten order.

    Returns tuple: (kdata_obj, trajectory)
    """
    n = size(acc_imag_data, 2)

    valid = trues(n)
    for i = 1:n
        if tfe_slot[i] < 0 || shot_repeat[i] < 0
            valid[i] = false
        end
    end
    n_invalid = count(!, valid)
    if n_invalid > 0
        @warn("Skipping $n_invalid/$n profile(s) with unresolved tfe_slot/shot_repeat")
    end

    num_tfe = maximum(tfe_slot[valid]) + 1
    num_shots = maximum(shot_repeat[valid]) + 1
    num_echoes = maximum(echo) + 1
    num_contr = maximum(contr[valid]) + 1
    num_chan = maximum(chan) + 1

    # Initialize 7D array: (kx, tfe, shots, echoes, contr, channels, interleaves=1)
    kdata = zeros(ComplexF32, num_kx, num_tfe, num_shots, num_echoes, num_contr, num_chan, 1)

    # Initialize trajectory: (2, tfe, shots, echoes, contr, interleaves=1)
    # profileOrder[1, :] = ky indices, profileOrder[2, :] = kz indices.
    # Default (unfilled) slots point at the first valid profile's (ky,kz);
    # since the matching kdata entry stays exactly zero, downstream
    # weighting (weightsMasked from dataTemp .== 0) excludes them from the
    # forward/adjoint operator regardless of this placeholder trajectory.
    profileOrder = zeros(Int, 2, num_tfe, num_shots, num_echoes, num_contr, 1)
    first_valid = findfirst(valid)
    if first_valid !== nothing
        profileOrder[1, :, :, :, :, :] .= ky[first_valid]
        profileOrder[2, :, :, :, :, :] .= kz[first_valid]
    end

    for i = 1:n
        if !valid[i]
            continue
        end
        tfe_idx = tfe_slot[i] + 1
        shot_idx = shot_repeat[i] + 1
        echo_idx = echo[i] + 1
        contr_idx = contr[i] + 1
        chan_idx = chan[i] + 1

        kdata[:, tfe_idx, shot_idx, echo_idx, contr_idx, chan_idx, 1] .= acc_imag_data[:, i]
        profileOrder[1, tfe_idx, shot_idx, echo_idx, contr_idx, 1] = ky[i]
        profileOrder[2, tfe_idx, shot_idx, echo_idx, contr_idx, 1] = kz[i]
    end

    println("[BUILD-CASPR] numTFE=$num_tfe numShots=$num_shots numEchoes=$num_echoes numContr=$num_contr numChan=$num_chan")

    return ReconBMRR.KdataPreprocessed(kdata), profileOrder
end

function maybe_debug_bridge_labels(ky, kz, echo, dyn, chan, extr1)
    BRIDGE_DEBUG_LABELS || return

    uniq_dyn = unique(dyn)
    uniq_chan = unique(chan)
    uniq_extr1 = unique(extr1)

    println("[BRIDGE-LABELS] unique counts: " *
            "ky=$(length(unique(ky))) kz=$(length(unique(kz))) " *
            "echo=$(length(unique(echo))) dyn=$(length(uniq_dyn)) " *
            "chan=$(length(uniq_chan)) extr1=$(length(uniq_extr1))")

    println("[BRIDGE-LABELS] dyn sample: " * string(collect(uniq_dyn[1:min(end, 12)])))
    println("[BRIDGE-LABELS] chan sample: " * string(collect(uniq_chan[1:min(end, 12)])))
    println("[BRIDGE-LABELS] extr1 sample: " * string(collect(uniq_extr1[1:min(end, 12)])))

    if BRIDGE_ENFORCE_EXPECTED_VIEWS
        counts = (length(uniq_dyn), length(uniq_chan), length(uniq_extr1))
        if BRIDGE_EXPECTED_VIEW_COUNT ∉ counts
            error("Expected view cardinality $(BRIDGE_EXPECTED_VIEW_COUNT) not found in dyn/chan/extr1 counts $(counts). " *
                  "Label assignment likely inconsistent with reference pipeline.")
        end
    end
end

function select_interleave_labels(
    extr1::Vector{UInt16},
    view_idx::Vector{Int32},
    dyn::Vector{UInt16},
    card::Vector{UInt16},
)
    uniq_extr1 = unique(extr1)
    uniq_dyn = unique(dyn)
    uniq_card = unique(card)
    uniq_view = isempty(view_idx) ? Int32[] : unique(view_idx)

    if BRIDGE_INTERLEAVE_SOURCE == "auto"
        if length(uniq_card) > 1 && length(uniq_extr1) == 1
            println("[BUILD] Auto interleave selection: using cardiac phase labels for Look-Locker semantics")
            return card, :card
        elseif !isempty(view_idx) && !any(view_idx .< 0)
            println("[BUILD] Auto interleave selection: using view_idx labels")
            return UInt16.(view_idx), :view_idx
        else
            println("[BUILD] Auto interleave selection: falling back to extr1 labels")
            return extr1, :extr1
        end
    elseif BRIDGE_INTERLEAVE_SOURCE == "view_idx"
        if isempty(view_idx)
            @warn "BRIDGE_INTERLEAVE_SOURCE=view_idx requested but view_idx is missing; using extr1"
            return extr1, :extr1
        elseif any(view_idx .< 0)
            @warn "BRIDGE_INTERLEAVE_SOURCE=view_idx requested but view_idx contains negative values; using extr1"
            return extr1, :extr1
        else
            return UInt16.(view_idx), :view_idx
        end
    elseif BRIDGE_INTERLEAVE_SOURCE == "dyn"
        return dyn, :dyn
    elseif BRIDGE_INTERLEAVE_SOURCE == "card"
        return card, :card
    elseif BRIDGE_INTERLEAVE_SOURCE == "extr1"
        return extr1, :extr1
    else
        @warn "Unknown BRIDGE_INTERLEAVE_SOURCE; using extr1" source=BRIDGE_INTERLEAVE_SOURCE
        return extr1, :extr1
    end
end

function validate_output_jld2(path::String)
    has_r_key = false
    loaded = nothing

    try
        loaded = jldopen(path, "r") do f
            obj = f["r"]
            return obj
        end
        has_r_key = true
    catch err
        error("Post-build validation failed to reload key r from output JLD2: $(err)")
    end

    type_ok = occursin("ReconParams", string(typeof(loaded)))
    acc_ok = hasproperty(loaded, :data) && hasproperty(loaded.data, :accImagData) && prod(size(loaded.data.accImagData)) > 0
    labels_ok = hasproperty(loaded, :data) && hasproperty(loaded.data, :labels) &&
                haskey(loaded.data.labels, :ky) && haskey(loaded.data.labels, :kz) &&
                haskey(loaded.data.labels, :chan) && haskey(loaded.data.labels, :dyn) &&
                haskey(loaded.data.labels, :echo)
    scan_ok = hasproperty(loaded, :scanParameters) && loaded.scanParameters isa AbstractDict &&
              haskey(loaded.scanParameters, :encodingSize) && haskey(loaded.scanParameters, :FieldStrength) &&
              haskey(loaded.scanParameters, :AcqMode)
    recon_ok = hasproperty(loaded, :reconParameters) && loaded.reconParameters isa AbstractDict &&
               haskey(loaded.reconParameters, :coilSensitivities) && haskey(loaded.reconParameters, :cudaSolver)

    phasecorr_cols = hasproperty(loaded.data, :phaseCorrData) ? size(loaded.data.phaseCorrData, 2) : 0
    noise_cols = hasproperty(loaded.data, :noiseData) && ndims(loaded.data.noiseData) >= 2 ? size(loaded.data.noiseData, 2) : 0
    acc_idx = Int.(vec(loaded.data.labels[:LabelLookupTable][1]))
    chan_first = unique(loaded.data.labels[:chan][acc_idx])[1]
    center_mask = (loaded.data.labels[:kz][acc_idx] .== 0) .&
                  (loaded.data.labels[:ky][acc_idx] .== 0) .&
                  (loaded.data.labels[:echo][acc_idx] .== 0) .&
                  (loaded.data.labels[:dyn][acc_idx] .== 0) .&
                  (loaded.data.labels[:chan][acc_idx] .== chan_first)
    center_repetitions = sum(center_mask)
    tfe_factor = haskey(loaded.scanParameters, :TFEfactor) ? Int(loaded.scanParameters[:TFEfactor]) : 1

    println("[POST-BUILD] key r loadable: " * string(has_r_key))
    println("[POST-BUILD] ReconParams type: " * string(type_ok))
    println("[POST-BUILD] accImagData present: " * string(acc_ok) * ", size=" * string(size(loaded.data.accImagData)))
    println("[POST-BUILD] required labels present: " * string(labels_ok))
    println("[POST-BUILD] required scan keys present: " * string(scan_ok))
    println("[POST-BUILD] required recon keys present: " * string(recon_ok))
    println("[POST-BUILD] phaseCorrData columns: " * string(phasecorr_cols))
    println("[POST-BUILD] noiseData columns: " * string(noise_cols))
    println("[POST-BUILD] center profile repetitions: " * string(center_repetitions) * ", TFEfactor=" * string(tfe_factor))

    if !(has_r_key && type_ok && acc_ok && labels_ok && scan_ok && recon_ok)
        error("Post-build validation failed: output JLD2 is not ready for ReconBMRR")
    end

    if phasecorr_cols == 0
        @warn("Output JLD2 has no phaseCorrData. recon.jl may need to skip phase correction or use a different input preparation path.")
    end
    if noise_cols == 0
        @warn("Output JLD2 has no noiseData/Psi calibration. recon.jl may need to skip noise prewhitening or use scanner-derived calibration.")
    end
    if tfe_factor > 1 && center_repetitions <= 1
        @warn("Output JLD2 has TFEfactor > 1 but only one center-profile repetition. ReconBMRR sortData() may fail because it infers shot structure from repeated ky=kz=0 profiles.")
    end
end

function usage_and_exit()
    println("Usage:")
    println("  julia scripts/build_jld2_from_mrtools_bridge.jl <bridge_npz> <output_jld2> [--template <template_jld2>] [--presort|--presort-caspr]")
    println("  julia scripts/build_jld2_from_mrtools_bridge.jl <bridge_npz> <template_jld2> <output_jld2> [--presort|--presort-caspr]   # legacy")
    println("")
    println("Options:")
    println("  --template       Optional template JLD2. If omitted, package-native defaults are used.")
    println("  --presort        Build KdataPreprocessed on a dense (ky,kz) Cartesian grid (skips sortData)")
    println("  --presort-caspr  Build KdataPreprocessed indexed by CASPR shot/TFE structure (tfe_slot,shot_repeat,contr),")
    println("                   required for CuCasprSubspaceOp-based subspace reconstruction")
    exit(1)
end

function parse_args(args::Vector{String})
    if length(args) < 2
        usage_and_exit()
    end

    bridge_npz = args[1]
    use_presort = false
    use_presort_caspr = false
    template_jld2 = nothing
    positional = String[]

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--presort"
            use_presort = true
            i += 1
        elseif arg == "--presort-caspr"
            use_presort = true
            use_presort_caspr = true
            i += 1
        elseif arg == "--template"
            if i == length(args)
                error("Missing value after --template")
            end
            template_jld2 = args[i + 1]
            i += 2
        else
            push!(positional, arg)
            i += 1
        end
    end

    if length(positional) == 1
        out_jld2 = positional[1]
    elseif length(positional) == 2 && isnothing(template_jld2)
        # Backward-compatible positional mode: <bridge_npz> <template_jld2> <output_jld2>
        template_jld2 = positional[1]
        out_jld2 = positional[2]
    else
        usage_and_exit()
    end

    return bridge_npz, template_jld2, out_jld2, use_presort, use_presort_caspr
end

function get_scalar(d::AbstractDict{String,<:Any}, key::String, default)
    if !haskey(d, key)
        return default
    end
    v = d[key]
    if v isa AbstractArray
        return length(v) == 0 ? default : v[1]
    end
    return v
end

function get_filename(d::AbstractDict{String,<:Any}, default::String)
    if !haskey(d, "filename_utf8")
        return default
    end
    raw = Vector{UInt8}(vec(d["filename_utf8"]))
    return String(raw)
end

function as_vector_int32(x)
    return Int32.(vec(x))
end

function as_vector_uint16(x)
    return UInt16.(vec(x))
end

function as_vector_int8(x)
    return Int8.(vec(x))
end

function get_echo_times_s(bridge::AbstractDict{String,<:Any}, n_echoes::Int, tr_raw::Float32)
    if haskey(bridge, "TE_s")
        te = Float32.(vec(bridge["TE_s"]))
        if length(te) >= n_echoes
            return te[1:n_echoes]
        end
    end

    if haskey(bridge, "te_s")
        te = Float32.(vec(bridge["te_s"]))
        if length(te) >= n_echoes
            return te[1:n_echoes]
        end
    end

    if haskey(bridge, "TE")
        te = Float32.(vec(bridge["TE"]))
        if length(te) >= n_echoes
            # Heuristic: TE values above 0.1 are likely milliseconds.
            return maximum(te) > 0.1f0 ? (te[1:n_echoes] .* 1f-3) : te[1:n_echoes]
        end
    end

    if haskey(bridge, "te")
        te = Float32.(vec(bridge["te"]))
        if length(te) >= n_echoes
            # Heuristic: TE values above 0.1 are likely milliseconds.
            return maximum(te) > 0.1f0 ? (te[1:n_echoes] .* 1f-3) : te[1:n_echoes]
        end
    end

    tr_s = tr_raw > 0.1f0 ? tr_raw * 1f-3 : tr_raw
    dt = tr_s > 0f0 ? max(tr_s / Float32(n_echoes + 1), 1f-3) : 1f-3
    return Float32.(collect(1:n_echoes) .* dt)
end

function normalize_vec3(v, default::Float32)
    if v isa AbstractArray
        vv = Float32.(vec(v))
        if length(vv) >= 3
            return vv[1:3]
        end
    end
    return Float32[default, default, default]
end

function get_encoding_size(scan::Dict{Symbol,Any}, bridge::AbstractDict{String,<:Any}, fallback::Vector{Int32}; prefer_existing::Bool=false)
    if prefer_existing && haskey(scan, :encodingSize)
        es = Int32.(vec(scan[:encodingSize]))
        if length(es) >= 3 && all(es[1:3] .> 0)
            return es[1:3]
        end
    end
    # NOTE (2026-09-14): recon_resolutions is grid-aligned (matches the k-space
    # grid this pipeline actually reconstructs on, including the nominal-grid
    # fix for the 200x75x141 discrepancy). scanner_recon_resolutions is the
    # scanner's own clinical/interpolated recon matrix (e.g. 128x96x101) and is
    # NOT the grid this custom minimal pipeline reconstructs on -- using it here
    # made encodingSize/FOV mismatch the actual reconstructed image size,
    # tripping the strict encoding-match check in recon_mrtools.jl. Prefer the
    # grid-aligned value; scanner_recon_resolutions is kept only for logging.
    if haskey(bridge, "recon_resolutions")
        rr = Int32.(vec(bridge["recon_resolutions"]))
        if length(rr) >= 3 && all(rr[1:3] .> 0)
            if haskey(bridge, "scanner_recon_resolutions")
                srr = Int32.(vec(bridge["scanner_recon_resolutions"]))
                if length(srr) >= 3 && srr[1:3] != rr[1:3]
                    @warn "scanner_recon_resolutions differs from recon_resolutions; preferring grid-aligned recon_resolutions" recon=rr[1:3] scanner=srr[1:3]
                end
            end
            return rr[1:3]
        end
    end
    if haskey(bridge, "scanner_recon_resolutions")
        srr = Int32.(vec(bridge["scanner_recon_resolutions"]))
        if length(srr) >= 3 && all(srr[1:3] .> 0)
            return srr[1:3]
        end
    end
    if haskey(scan, :encodingSize)
        es = Int32.(vec(scan[:encodingSize]))
        if length(es) >= 3 && all(es[1:3] .> 0)
            return es[1:3]
        end
    end
    return Int32.(fallback)
end

bridge_npz, template_jld2, out_jld2, use_presort, use_presort_caspr = parse_args(ARGS)
bridge = NPZ.npzread(bridge_npz)
has_template = !isnothing(template_jld2)
r_template = has_template ? jldopen(template_jld2)["r"] : nothing

acc_imag_data = Array{ComplexF32}(bridge["accImagData"])
num_kx, n = size(acc_imag_data)

ky = as_vector_int32(bridge["ky"])
kz = as_vector_int32(bridge["kz"])
echo = as_vector_uint16(bridge["echo"])
dyn = as_vector_uint16(bridge["dyn"])
chan = as_vector_uint16(bridge["chan"])
extr1 = as_vector_uint16(bridge["extr1"])
card = as_vector_uint16(bridge["card"])
sign = as_vector_int8(bridge["sign"])
typ = UInt8.(vec(bridge["typ"]))
mix = as_vector_uint16(bridge["mix"])
seq_nr = haskey(bridge, "seq_nr") ? as_vector_int32(bridge["seq_nr"]) : Int32[]
view_idx = haskey(bridge, "view_idx") ? as_vector_int32(bridge["view_idx"]) : Int32[]
extr1_original = haskey(bridge, "extr1_original") ? as_vector_int32(bridge["extr1_original"]) : Int32[]
tfe_slot = haskey(bridge, "tfe_slot") ? as_vector_int32(bridge["tfe_slot"]) : Int32[]
shot_repeat = haskey(bridge, "shot_repeat") ? as_vector_int32(bridge["shot_repeat"]) : Int32[]
contr = haskey(bridge, "contr") ? as_vector_int32(bridge["contr"]) : Int32[]

extr1_selected, interleave_source_used = select_interleave_labels(extr1, view_idx, dyn, card)
println("[BUILD] Interleave source used: " * String(interleave_source_used))

maybe_debug_bridge_labels(ky, kz, echo, dyn, chan, extr1_selected)

if length(ky) != n || length(kz) != n || length(echo) != n || length(dyn) != n ||
   length(chan) != n || length(extr1_selected) != n || length(card) != n || length(sign) != n ||
   length(typ) != n || length(mix) != n
    error("Bridge arrays do not have consistent length n=$n")
end

if !isempty(seq_nr) && length(seq_nr) != n
    error("Bridge field seq_nr has inconsistent length $(length(seq_nr)) != n=$n")
end
if !isempty(view_idx) && length(view_idx) != n
    error("Bridge field view_idx has inconsistent length $(length(view_idx)) != n=$n")
end
if !isempty(extr1_original) && length(extr1_original) != n
    error("Bridge field extr1_original has inconsistent length $(length(extr1_original)) != n=$n")
end

if use_presort_caspr
    if isempty(tfe_slot) || isempty(shot_repeat) || isempty(contr)
        error("--presort-caspr requires bridge fields tfe_slot, shot_repeat, contr (re-export with an updated export_reconbmrr_bridge.py)")
    end
    if length(tfe_slot) != n || length(shot_repeat) != n || length(contr) != n
        error("Bridge fields tfe_slot/shot_repeat/contr have inconsistent length != n=$n")
    end
end

scan = build_scan_parameters_base()

# Build data object based on presort flag
if use_presort_caspr
    println("[BUILD] Using CASPR shot/TFE-indexed KdataPreprocessed path (skips sortData, targets CuCasprSubspaceOp)")
    kdata, profileOrder = build_kdatapreprocessed_caspr_subspace_from_bridge(acc_imag_data, ky, kz, echo, tfe_slot, shot_repeat, contr, chan, num_kx)
    labels = Dict{Symbol, Any}()  # Empty for preprocessed, we embed info in the 7D array structure
elseif use_presort
    println("[BUILD] Using pre-sorted KdataPreprocessed path (skips sortData)")
    # Use the nominal (sequence-designed) ky/kz encoding matrix bounds from
    # the .sin header when available, rather than shrinking the grid to only
    # this scan's acquired extent (see TODO.md 2026-09-14 "200x75x141
    # k-space dimensions" reviewer finding). NOTE: the bridge exports a
    # 1-element sentinel value (-1000000), NOT a 0-length array, to signal
    # "absent" -- a 0-length 1D int32 array positioned after the multi-GB
    # accImagData entry triggers a reproducible EOFError in Julia's
    # NPZ.jl/ZipFile.jl (see TODO.md 2026-09-14 "NPZ.jl zero-length array" entry).
    nominal_grid_sentinel = -1000000
    read_nominal_bound(bridge, key) = haskey(bridge, key) ? (v = Int(only(bridge[key])); v == nominal_grid_sentinel ? nothing : v) : nothing
    nom_ky_min = read_nominal_bound(bridge, "nominal_ky_min")
    nom_ky_max = read_nominal_bound(bridge, "nominal_ky_max")
    nom_kz_min = read_nominal_bound(bridge, "nominal_kz_min")
    nom_kz_max = read_nominal_bound(bridge, "nominal_kz_max")
    kdata, profileOrder = build_kdatapreprocessed_from_bridge(acc_imag_data, ky, kz, echo, dyn, chan, extr1_selected, num_kx;
        nominal_ky_min=nom_ky_min, nominal_ky_max=nom_ky_max, nominal_kz_min=nom_kz_min, nominal_kz_max=nom_kz_max)
    labels = Dict{Symbol, Any}()  # Empty for preprocessed, we embed info in the 7D array structure
else
    profileOrder = nothing
    labels = Dict{Symbol,Any}(
        :ky => ky,
        :kz => kz,
        :echo => echo,
        :dyn => dyn,
        :chan => chan,
        :extr1 => extr1_selected,
        :card => card,
        :sign => sign,
        :typ => typ,
        :mix => mix,
        :LabelLookupTable => build_label_lookup_table(n),
    )
    if !isempty(seq_nr)
        labels[:seq_nr] = seq_nr
    end
    if !isempty(view_idx)
        labels[:view_idx] = view_idx
    end
    if !isempty(extr1_original)
        labels[:extr1_original] = extr1_original
    end

    rej_imag_data = zeros(ComplexF32, num_kx, 0)
    phase_corr_data = zeros(ComplexF32, num_kx, 0)
    freq_corr_data = zeros(ComplexF32, num_kx, 0)
    noise_data = zeros(ComplexF32, 0, 0)

    kdata = ReconBMRR.KdataRaw{Float32}(
        acc_imag_data,
        rej_imag_data,
        phase_corr_data,
        freq_corr_data,
        noise_data,
        labels,
    )
end

if has_template
    scan = deep_merge(scan, r_template.scanParameters)
end

ky_min = Int32(get_scalar(bridge, "ky_min", minimum(ky)))
ky_max = Int32(get_scalar(bridge, "ky_max", maximum(ky)))
kz_min = Int32(get_scalar(bridge, "kz_min", minimum(kz)))
kz_max = Int32(get_scalar(bridge, "kz_max", maximum(kz)))

scan[:KyRange] = collect(ky_min:ky_max)
scan[:KzRange] = collect(kz_min:kz_max)
scan[:KxRange] = collect(Int32(0):Int32(num_kx - 1))
scan[:TFEfactor] = max(1, Int(get_scalar(bridge, "tfe_factor", 1)))
scan[:encodingSize] = get_encoding_size(
    scan,
    bridge,
    Int32[num_kx, ky_max - ky_min + 1, kz_max - kz_min + 1];
    prefer_existing=has_template,
)
scan[:AcqMode] = "Cartesian"

scan[:AcqVoxelSize] = normalize_vec3(get(scan, :AcqVoxelSize, Float32[]), 1.0f0)
scan[:RecVoxelSize] = normalize_vec3(get(scan, :RecVoxelSize, Float32[]), 1.0f0)
fov = normalize_vec3(get(scan, :FOV, Float32[]), 0.0f0)
expected_encoding = Int32.(round.(Int, fov ./ scan[:RecVoxelSize]))
if any(fov .<= 0.0f0) || any(abs.(expected_encoding .- scan[:encodingSize]) .> 1)
    # Keep FOV and encodingSize consistent to prevent oversampling crop to 1x1x1.
    fov = Float32.(scan[:encodingSize]) .* scan[:RecVoxelSize]
end
scan[:FOV] = fov

tr = Float32(get_scalar(bridge, "tr", scan[:TR]))
flip = Float32(get_scalar(bridge, "flip_angle", scan[:FlipAngle]))
scan[:TR] = tr
scan[:FlipAngle] = flip

n_echoes = max(1, Int(maximum(echo)) + 1)
te_s = get_echo_times_s(bridge, n_echoes, tr)
scan[:TE_s] = te_s
scan[:TE] = te_s

if !haskey(scan, :centerFreq_Hz)
    b0 = haskey(scan, :FieldStrength) ? Float32(scan[:FieldStrength]) : 3.0f0
    # Proton gyromagnetic ratio in Hz/T.
    scan[:centerFreq_Hz] = Float32(42.57747892f6) * b0
end

recon = build_recon_parameters_base(scan)

# Marks CASPR shot/TFE-indexed kdata (axes 2/3 are temporal tfe/shot, not
# spatial ky/kz) so recon_mrtools.jl's refresh_recon_state! knows NOT to
# recompute sensMaps via a spatial iFFT over those axes (see TODO.md
# "THIRD latent bug found before GPU submission").
recon[:casprSubspaceIndexed] = use_presort_caspr
println("[BUILD] casprSubspaceIndexed = " * string(recon[:casprSubspaceIndexed]))

# Noise pre-whitening: use this scan's own noise covariance if the bridge NPZ
# provides one (see experiment/TODO.md 2026-09-14 "noise pre-whitening"); a
# stale/mismatched Psi from a template must never be silently reused, since
# noisePreWhitening! requires Psi to match this build's own channel count.
if haskey(bridge, "noise_covariance") && length(bridge["noise_covariance"]) > 0
    psi = ComplexF32.(bridge["noise_covariance"])
    expected_num_chan = Int(maximum(chan)) + 1
    if size(psi, 1) == size(psi, 2) == expected_num_chan
        scan[:Psi] = psi
        println("[BUILD] Psi (noise covariance) set: size=" * string(size(psi)))
    else
        @warn "noise_covariance shape $(size(psi)) does not match numChan=$expected_num_chan; skipping Psi."
        delete!(scan, :Psi)
    end
elseif haskey(scan, :Psi)
    delete!(scan, :Psi)
end

# Bipolar echo-misalignment correction: fit the phase-ramp model from this
# scan's own calibration lines (see experiment/TODO.md 2026-09-14 "bipolar
# phase correction"). phaseCorrDataBipolar! only works on KdataRaw, but our
# --presort/--presort-caspr paths build KdataPreprocessed directly, so we
# build a minimal throwaway KdataRaw just to run the fit, then carry the
# fitted params (not the raw calibration data itself) into the real recon.
if haskey(bridge, "phase_corr_data") && size(bridge["phase_corr_data"], 2) > 0
    pcd = ComplexF32.(bridge["phase_corr_data"])
    pcd_echo = as_vector_int32(bridge["phase_corr_echo"])
    pcd_sign = as_vector_int8(bridge["phase_corr_sign"])
    ncalib = size(pcd, 2)
    calib_labels = Dict{Symbol,Any}(
        :echo => UInt16.(pcd_echo),
        :sign => pcd_sign,
        :LabelLookupTable => Any[reshape(collect(1:ncalib), 1, :), reshape(Float64[], 1, 0),
                                  reshape(collect(1:ncalib), 1, :), reshape(Float64[], 1, 0), reshape(Float64[], 1, 0)],
    )
    calib_data = ReconBMRR.KdataRaw(
        pcd, zeros(ComplexF32, size(pcd, 1), 0), pcd, zeros(ComplexF32, size(pcd, 1), 0),
        zeros(ComplexF32, size(pcd, 1), 0), calib_labels,
    )
    calib_r = ReconBMRR.ReconParams(
        "phase_corr_calib", ".", Dict{Symbol,Any}(), Dict{Symbol,Any}(), calib_data,
        ReconBMRR.Cartesian3D(zeros(Int, 2, 1, 1, 1, 1, 1), :Cartesian3D), Symbol[], nothing,
    )
    try
        ReconBMRR.phaseCorrDataBipolar!(calib_r)
        recon[:phaseCorrDataBipolar] = calib_r.reconParameters[:phaseCorrDataBipolar]
        println("[BUILD] phaseCorrDataBipolar fit: " * string(recon[:phaseCorrDataBipolar]))
    catch err
        @warn "Bipolar phase-correction fit failed; proceeding without it." exception=err
    end
end

# For preprocessed data, we have pre-built trajectory; otherwise create minimal
if use_presort
    traj = ReconBMRR.Cartesian3D(profileOrder, :Cartesian3D)
else
    traj = ReconBMRR.Cartesian3D(zeros(Int, 2, 1, 1, 1, 1, 1), :Cartesian3D)
end

filename = get_filename(bridge, "mrtools_bridge")
path_proc = dirname(filename)

r = ReconBMRR.ReconParams(
    filename,
    path_proc,
    scan,
    recon,
    kdata,
    traj,
    Symbol[],
    nothing,
)

if has_template
    r.reconParameters = deep_merge(recon, r_template.reconParameters)
    # deep_merge lets the template's own reconParameters override ours, but
    # casprSubspaceIndexed must always reflect THIS build's own kdata layout,
    # never a template's (e.g. a grid-indexed self-calibration template would
    # otherwise clobber it back to false and silently re-trigger the spatial
    # sensMaps recompute in recon_mrtools.jl's refresh_recon_state!).
    r.reconParameters[:casprSubspaceIndexed] = use_presort_caspr
    # Same reasoning for phaseCorrDataBipolar: it's a fit computed fresh from
    # THIS scan's own calibration lines and must never be silently clobbered
    # or inherited from an unrelated template.
    if haskey(recon, :phaseCorrDataBipolar)
        r.reconParameters[:phaseCorrDataBipolar] = recon[:phaseCorrDataBipolar]
    elseif haskey(r.reconParameters, :phaseCorrDataBipolar)
        delete!(r.reconParameters, :phaseCorrDataBipolar)
    end
    if !use_presort
        r.data.labels = deep_merge(r_template.data.labels, r.data.labels)
    end
    println("[BUILD] Using template overrides from: " * string(template_jld2))
else
    println("[BUILD] No template provided; using package-native defaults")
end

if use_presort
    println("[BUILD] Writing pre-sorted JLD2 with compression over IOStream")
    jldsave(out_jld2, true, IOStream; r=r)
else
    jldsave(out_jld2; r=r)
end

if !use_presort
    validate_output_jld2(out_jld2)
else
    jldopen(out_jld2, "r") do f
        f["r"]
    end
    println("[POST-BUILD] Reloaded pre-sorted JLD2 successfully")
    println("[POST-BUILD] KdataPreprocessed shape: ", size(kdata.kdata))
    println("[POST-BUILD] Ready for reconstruction (no sortData needed)")
end

println("Wrote JLD2: " * out_jld2)
if use_presort
    println("Using pre-sorted KdataPreprocessed")
else
    println("accImagData size: " * string(size(r.data.accImagData)))
    println("KyRange: " * string((minimum(r.scanParameters[:KyRange]), maximum(r.scanParameters[:KyRange]))))
    println("KzRange: " * string((minimum(r.scanParameters[:KzRange]), maximum(r.scanParameters[:KzRange]))))
end
