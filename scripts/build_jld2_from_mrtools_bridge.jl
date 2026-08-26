using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using NPZ
using JLD2
using ReconBMRR

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
    num_kx::Int,
)
    """
    Build KdataPreprocessed directly from MRTOOLS bridge data, bypassing sortData().
    Returns tuple: (kdata_obj, trajectory)
    """
    
    # Get unique values and ranges
    ky_min = minimum(ky)
    ky_max = maximum(ky)
    kz_min = minimum(kz)
    kz_max = maximum(kz)
    
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
    println("  julia scripts/build_jld2_from_mrtools_bridge.jl <bridge_npz> <output_jld2> [--template <template_jld2>] [--presort]")
    println("  julia scripts/build_jld2_from_mrtools_bridge.jl <bridge_npz> <template_jld2> <output_jld2> [--presort]   # legacy")
    println("")
    println("Options:")
    println("  --template   Optional template JLD2. If omitted, package-native defaults are used.")
    println("  --presort    Build KdataPreprocessed instead of KdataRaw (skips sortData)")
    exit(1)
end

function parse_args(args::Vector{String})
    if length(args) < 2
        usage_and_exit()
    end

    bridge_npz = args[1]
    use_presort = false
    template_jld2 = nothing
    positional = String[]

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--presort"
            use_presort = true
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

    return bridge_npz, template_jld2, out_jld2, use_presort
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

bridge_npz, template_jld2, out_jld2, use_presort = parse_args(ARGS)
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

if length(ky) != n || length(kz) != n || length(echo) != n || length(dyn) != n ||
   length(chan) != n || length(extr1) != n || length(card) != n || length(sign) != n ||
   length(typ) != n || length(mix) != n
    error("Bridge arrays do not have consistent length n=$n")
end

scan = build_scan_parameters_base()

# Build data object based on presort flag
if use_presort
    println("[BUILD] Using pre-sorted KdataPreprocessed path (skips sortData)")
    kdata, profileOrder = build_kdatapreprocessed_from_bridge(acc_imag_data, ky, kz, echo, dyn, chan, extr1, num_kx)
    labels = Dict{Symbol, Any}()  # Empty for preprocessed, we embed info in the 7D array structure
else
    profileOrder = nothing
    labels = Dict{Symbol,Any}(
        :ky => ky,
        :kz => kz,
        :echo => echo,
        :dyn => dyn,
        :chan => chan,
        :extr1 => extr1,
        :card => card,
        :sign => sign,
        :typ => typ,
        :mix => mix,
        :LabelLookupTable => build_label_lookup_table(n),
    )

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
scan[:encodingSize] = Int32[num_kx, ky_max - ky_min + 1, kz_max - kz_min + 1]
scan[:AcqMode] = "Cartesian"

tr = Float32(get_scalar(bridge, "tr", scan[:TR]))
flip = Float32(get_scalar(bridge, "flip_angle", scan[:FlipAngle]))
scan[:TR] = tr
scan[:FlipAngle] = flip

recon = build_recon_parameters_base(scan)

# Skip prewhitening in bridge recon unless a valid Psi is explicitly provided.
if haskey(scan, :Psi)
    delete!(scan, :Psi)
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
    if !use_presort
        r.data.labels = deep_merge(r_template.data.labels, r.data.labels)
    end
    println("[BUILD] Using template overrides from: " * string(template_jld2))
else
    println("[BUILD] No template provided; using package-native defaults")
end

jldsave(out_jld2; r=r)

if !use_presort
    validate_output_jld2(out_jld2)
else
    println("[POST-BUILD] Skipping full validation for pre-sorted data (structure is different)")
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
