# Debug variant of recon_mrtools.jl: identical pipeline logic, with inspect()
# checkpoints added and everything after iterativeRecon (masking/upsample/export)
# removed. Mirrors bridge_v2/abdominal-t1t2-mapping/scripts/recon_debug.jl's
# print format so the two runs can be compared side by side.
# Run with `julia --project=. -i scripts/recon_mrtools_debug.jl data/phantom/dixon_caspr_calib74_with_3_5mmV4_raw_latest.jld2 src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5 --bypass-sort`
using Pkg
Pkg.activate(".") # load the environment from Project.toml
using Revise
using ReconBMRR
using RegularizedLeastSquares
using JLD2, CodecZlib
using FFTW
using HDF5
using Statistics

const RECON_USE_CUDA = get(ENV, "RECON_USE_CUDA", "0") == "1"
const TRACE_SORTDATA = get(ENV, "RECON_TRACE_SORTDATA", "0") == "1"
const STRICT_ENCODING_MATCH = get(ENV, "RECON_STRICT_ENCODING_MATCH", "1") == "1"
const FORCE_CPU_SOLVER = get(ENV, "RECON_FORCE_CPU_SOLVER", "0") == "1"
const RECON_ADMM_ITERATIONS =
    try
        max(1, parse(Int, get(ENV, "RECON_ADMM_ITERATIONS", "15")))
    catch
        15
    end
const RECON_CG_ITERATIONS =
    try
        max(1, parse(Int, get(ENV, "RECON_CG_ITERATIONS", "10")))
    catch
        10
    end
const REQUESTED_MAX_CHANNELS = try
    parse(Int, get(ENV, "RECON_MAX_CHANNELS", "0"))
catch
    0
end
const OOM_RETRY_MAX_CHANNELS = try
    parse(Int, get(ENV, "RECON_OOM_RETRY_MAX_CHANNELS", "8"))
catch
    8
end
const MAX_ENCODING_REL_DIFF = try
    parse(Float64, get(ENV, "RECON_MAX_ENCODING_REL_DIFF", "0.10"))
catch
    0.10
end

function init_cuda_available()
    if !RECON_USE_CUDA
        @info "CUDA disabled for reconstruction; running CPU-only mode"
        return false
    end

    try
        @eval using CUDA
        CUDA.device!(0)
        @info "CUDA initialized for reconstruction" device=CUDA.name(CUDA.device())
        return true
    catch err
        @warn "CUDA initialization failed; continuing in CPU-only mode" exception=(err, catch_backtrace())
        return false
    end
end

const CUDA_AVAILABLE = init_cuda_available()
println("[BACKEND] RECON_USE_CUDA=$(RECON_USE_CUDA) CUDA_AVAILABLE=$(CUDA_AVAILABLE)")
flush(stdout)

# Same print format as ReconBMRR.summarize_dataset (bridge_v2/src/Export.jl) so
# the two debug runs are directly comparable.
function inspect(label::AbstractString, x)
    if x isa AbstractArray && !isempty(x)
        println("  ", label, ": size=", size(x), " eltype=", eltype(x))
        mag = eltype(x) <: Complex ? abs.(x) : x
        if eltype(mag) <: Real
            println("    range=[", minimum(mag), ", ", maximum(mag), "] mean=", sum(mag) / length(mag))
        end
        nBad = count(!isfinite, mag)
        nBad > 0 && println("    WARNING: ", nBad, " non-finite values")
    else
        println("  ", label, ": ", x)
    end
    flush(stdout)
end

function inspect_reconparams(r, keys)
    for k in keys
        haskey(r.reconParameters, k) && inspect(string("reconParameters[:", k, "]"), r.reconParameters[k])
    end
end

function should_fail_geometry(target_encoding::AbstractVector{<:Integer}, spatial_size::AbstractVector{<:Integer})
    target_f = Float64.(target_encoding)
    spatial_f = Float64.(spatial_size)
    expand_rel = (target_f .- spatial_f) ./ max.(1.0, spatial_f)
    return any(target_encoding .<= 1) || any(expand_rel .> MAX_ENCODING_REL_DIFF)
end

function resolve_dict_path()
    if length(ARGS) >= 2
        return ARGS[2]
    end
    if haskey(ENV, "RECON_DICT_PATH")
        return ENV["RECON_DICT_PATH"]
    end
    return "src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5"
end

const DICT_PATH = resolve_dict_path()
@info "Using dictionary path: $DICT_PATH"

function radial_calib_taper(ny::Int, nz::Int, r_inner::Float64, r_outer::Float64)
    w = ones(Float64, ny, nz)
    cy, cz = (ny - 1) / 2, (nz - 1) / 2
    for j in 1:nz, i in 1:ny
        r = sqrt((i - 1 - cy)^2 + (j - 1 - cz)^2)
        if r <= r_inner
            w[i, j] = 1.0
        elseif r >= r_outer
            w[i, j] = 0.0
        else
            w[i, j] = 0.5 * (1 + cos(pi * (r - r_inner) / (r_outer - r_inner)))
        end
    end
    return w
end

function compute_sensmaps_sos(kdata::Array{Complex{T}, 7}) where T<:AbstractFloat
    kx, ky, kz, necho, ndyn, nchan, ninter = size(kdata)
    kdata_avg = dropdims(mean(kdata, dims=(4, 5, 7)), dims=(4, 5, 7))
    taper = Float32.(radial_calib_taper(ky, kz, 6.0, 10.0))
    kdata_avg = kdata_avg .* reshape(taper, 1, ky, kz, 1)
    imgs = similar(kdata_avg)
    for c in 1:nchan
        kspace_c = kdata_avg[:, :, :, c]
        imgs[:, :, :, c] .= fftshift(ifft(ifftshift(kspace_c, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))
    end
    sos = sqrt.(sum(abs.(imgs).^2, dims=4) .+ eps(T))
    sensmaps = imgs ./ sos
    return sensmaps
end

function resolved_channel_limit(requested::Integer, available::Integer)
    if requested <= 0 || requested >= available
        return available
    end
    return max(1, Int(requested))
end

function refresh_recon_state!(r)
    kd = r.data.kdata
    r.reconParameters[:numKx] = size(kd, 1)
    r.reconParameters[:numKy] = size(kd, 2)
    r.reconParameters[:numKz] = size(kd, 3)

    caspr_indexed = get(r.reconParameters, :casprSubspaceIndexed, false)

    if haskey(r.reconParameters, :sensMaps)
        sm = r.reconParameters[:sensMaps]
        if caspr_indexed
            if size(sm, 4) != size(kd, 6)
                error("sensMaps channel count $(size(sm,4)) does not match kdata channel count $(size(kd,6)) " *
                      "for CASPR shot/TFE-indexed data. A template with matching channel-count sensMaps is required " *
                      "(spatial sensMaps cannot be recomputed from temporally-indexed kdata).")
            end
        elseif size(sm)[1:3] != size(kd)[1:3] || size(sm, 4) != size(kd, 6)
            @warn "sensMaps $(size(sm)) do not match kdata (spatial=$(size(kd)[1:3]), chan=$(size(kd,6))). " *
                  "Recomputing sensMaps from kdata using sum-of-squares method."
            r.reconParameters[:sensMaps] = compute_sensmaps_sos(kd)
            @info "New sensMaps size: $(size(r.reconParameters[:sensMaps]))"
        end
    elseif caspr_indexed
        error("No sensMaps in JLD2 for CASPR shot/TFE-indexed data, and sensMaps cannot be computed from " *
              "temporally-indexed kdata via compute_sensmaps_sos. Provide a template JLD2 with valid sensMaps.")
    else
        @info "No sensMaps in JLD2. Computing from kdata using sum-of-squares method."
        r.reconParameters[:sensMaps] = compute_sensmaps_sos(kd)
        @info "New sensMaps size: $(size(r.reconParameters[:sensMaps]))"
    end

    computeDensityCompensation!(r)
end

function apply_channel_limit!(r, requested::Integer; reason::AbstractString="")
    available = size(r.data.kdata, 6)
    target = resolved_channel_limit(requested, available)
    if target == available
        return false
    end

    @warn "Reducing channel count for lower-memory reconstruction" reason=reason requested=requested selected=target available=available
    r.data.kdata = r.data.kdata[:, :, :, :, :, 1:target, :]
    if haskey(r.reconParameters, :sensMaps)
        if get(r.reconParameters, :casprSubspaceIndexed, false)
            sm = r.reconParameters[:sensMaps]
            r.reconParameters[:sensMaps] = sm[:, :, :, 1:target]
        else
            delete!(r.reconParameters, :sensMaps)
        end
    end
    if haskey(r.reconParameters, :sdcCartesian)
        delete!(r.reconParameters, :sdcCartesian)
    end
    refresh_recon_state!(r)
    @info "Reduced-channel reconstruction state prepared" kdata_size=size(r.data.kdata) sensmaps_size=size(r.reconParameters[:sensMaps])
    return true
end

function oom_retry_channel_limit(r)
    current = size(r.data.kdata, 6)
    target = resolved_channel_limit(OOM_RETRY_MAX_CHANNELS, current)
    if target < current
        return target
    end
    return max(1, fld(current, 2))
end

filename = length(ARGS) >= 1 ? ARGS[1] : "data/phantom/dixon_caspr_calib74_with_3_5mmV4_raw_latest.jld2"
bypass_sort = "--bypass-sort" in ARGS
motionCorrection = false

println("=== raw load ===")
flush(stdout)
r = jldopen(filename)["r"]
r.pathProc = dirname(replace(filename, "exp_raw" => "exp_pro"))
isdir(r.pathProc) || mkpath(r.pathProc)
inspect("r.data.kdata (raw)", hasproperty(r.data, :accImagData) ? r.data.accImagData : r.data.kdata)
println("scanParameters keys: ", collect(keys(r.scanParameters)))
println("TE_s=", get(r.scanParameters, :TE_s, missing), " FOV=", get(r.scanParameters, :FOV, missing),
        " AcqVoxelSize=", get(r.scanParameters, :AcqVoxelSize, missing))

has_phase_corr = hasproperty(r.data, :phaseCorrData) && size(r.data.phaseCorrData, 2) > 0
has_noise_cov = haskey(r.scanParameters, :Psi)
is_preprocessed = isa(r.data, ReconBMRR.KdataPreprocessed) || bypass_sort

@info "Input data type: $(typeof(r.data).name.name)"
@info "Is preprocessed: $is_preprocessed"

if has_phase_corr
    if is_preprocessed
        @warn "Phase correction data present but data is already preprocessed; skipping phaseCorrDataBipolar!"
    else
        phaseCorrDataBipolar!(r)
    end
else
    @info "Skipping phaseCorrDataBipolar!: no phase correction data present in input JLD2"
end

println("=== preprocessing ===")
flush(stdout)
if isa(r.data, ReconBMRR.KdataPreprocessed)
    @info "Data is already KdataPreprocessed, skipping sortData()"
    r2 = r
elseif bypass_sort
    @info "--bypass-sort: converting KdataRaw → KdataPreprocessed without sortData()"
    labels = r.data.labels
    acc = r.data.accImagData
    num_kx = size(acc, 1)
    n = size(acc, 2)
    acc_idx = Int.(vec(labels[:LabelLookupTable][1]))
    ky_v = labels[:ky][acc_idx]
    kz_v = labels[:kz][acc_idx]
    echo_v = labels[:echo][acc_idx]
    dyn_v  = labels[:dyn][acc_idx]
    chan_v  = labels[:chan][acc_idx]
    extr1_v = labels[:extr1][acc_idx]
    ky_min = minimum(ky_v);  ky_max = maximum(ky_v)
    kz_min = minimum(kz_v);  kz_max = maximum(kz_v)
    num_ky = ky_max - ky_min + 1
    num_kz = kz_max - kz_min + 1
    num_echoes = Int(maximum(echo_v)) + 1
    num_dyn    = Int(maximum(dyn_v))  + 1
    num_chan   = Int(maximum(chan_v))  + 1
    num_inter  = Int(maximum(extr1_v)) + 1
    kdata7 = zeros(ComplexF32, num_kx, num_ky, num_kz, num_echoes, num_dyn, num_chan, num_inter)
    for i in 1:length(acc_idx)
        col = acc_idx[i]
        kdata7[:, Int(ky_v[i])-ky_min+1, Int(kz_v[i])-kz_min+1,
               Int(echo_v[i])+1, Int(dyn_v[i])+1, Int(chan_v[i])+1, Int(extr1_v[i])+1] .= acc[:, col]
    end
    kdata_pre = ReconBMRR.KdataPreprocessed(kdata7)
    profileOrder = zeros(Int, 2, num_ky, num_kz, num_echoes, num_dyn, num_inter)
    for i in 1:length(acc_idx)
        col = acc_idx[i]
        ki = Int(ky_v[i]) - ky_min + 1
        zi = Int(kz_v[i]) - kz_min + 1
        ei = Int(echo_v[i]) + 1
        di = Int(dyn_v[i]) + 1
        ii = Int(extr1_v[i]) + 1
        profileOrder[1, ki, zi, ei, di, ii] = ki
        profileOrder[2, ki, zi, ei, di, ii] = zi
    end
    new_traj = ReconBMRR.Cartesian3D(profileOrder, :Cartesian3D)
    r2 = ReconBMRR.ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters,
                               kdata_pre, new_traj, r.performedMethods, r.imgData)
else
    labels_check = r.data.labels
    acc_idx_check = Int.(vec(labels_check[:LabelLookupTable][1]))
    chan_first_check = unique(labels_check[:chan][acc_idx_check])[1]
    center_mask_check = (labels_check[:kz][acc_idx_check] .== 0) .&
                        (labels_check[:ky][acc_idx_check] .== 0) .&
                        (labels_check[:echo][acc_idx_check] .== 0) .&
                        (labels_check[:dyn][acc_idx_check] .== 0) .&
                        (labels_check[:chan][acc_idx_check] .== chan_first_check)
    center_reps = sum(center_mask_check)
    if center_reps <= 1
        error("sortData() requires repeated center (ky=kz=0) profiles to infer shot structure, " *
              "but only $center_reps center profile(s) found. " *
              "MRTOOLS bridge data does not carry this structure — use --bypass-sort instead.")
    end
    @info "Data is KdataRaw, applying sortData()"
    r2 = sortData(r)
end
inspect("r2.data.kdata (after sort/bypass-sort)", r2.data.kdata)

if haskey(r2.reconParameters, :phaseCorrDataBipolar)
    applyPhaseCorrDataBipolar!(r2)
else
    @info "Skipping applyPhaseCorrDataBipolar!: no phase correction fit was computed"
end

if has_noise_cov
    noisePreWhitening!(r2)
else
    @info "Skipping noisePreWhitening!: no Psi noise covariance matrix present in input JLD2"
end

needs_interleave_collapse = typeof(r2.traj) == ReconBMRR.Cartesian3D && ndims(r2.data.kdata) == 7 && size(r2.data.kdata, 7) > 1
if needs_interleave_collapse
    @info "Collapsing interleaves into dynamics for Cartesian reconstruction" dyn=size(r2.data.kdata, 5) interleaves=size(r2.data.kdata, 7)
    changeInterleavesToDynamics!(r2)
else
    @info "Skipping changeInterleavesToDynamics!: data is already preprocessed or bypass-sort was used"
end
inspect("r2.data.kdata (after changeInterleavesToDynamics)", r2.data.kdata)

r2.reconParameters[:cuda] = CUDA_AVAILABLE
r2.reconParameters[:cudaSolver] = CUDA_AVAILABLE
r2.reconParameters[:motionStatesRecon] = 1
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] = 0.001
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] = 0.0
r2.reconParameters[:iterativeReconParams][:Regularization][:LLR] = 0.0
r2.reconParameters[:iterativeReconParams][:subspaceRecon] = true
r2.reconParameters[:iterativeReconParams][:subspaceComponents] = 5
r2.reconParameters[:iterativeReconParams][:iterations] = RECON_ADMM_ITERATIONS
r2.reconParameters[:iterativeReconParams][:iterationsCG] = RECON_CG_ITERATIONS
r2.reconParameters[:iterativeReconParams][:vary_rho] = :balance
r2.reconParameters[:iterativeReconParams][:rho] = 0.01
r2.reconParameters[:prepDictPath] = DICT_PATH

if FORCE_CPU_SOLVER
    @warn "RECON_FORCE_CPU_SOLVER=1: forcing CPU solver while keeping CUDA availability for non-solver operations"
    r2.reconParameters[:cudaSolver] = false
end

if motionCorrection
    softGatingWeights!(r2)
end

apply_channel_limit!(r2, REQUESTED_MAX_CHANNELS; reason="preconfigured low-memory mode")
subspaceBasis!(r2)
refresh_recon_state!(r2)
inspect_reconparams(r2, [:subspaceBasis, :sensMaps, :sdcCartesian])

function is_gpu_oom(err)
    msg = sprint(showerror, err)
    return occursin("Out of GPU memory", msg) ||
           occursin("out of memory", lowercase(msg)) ||
           occursin("ERROR_OUT_OF_MEMORY", msg)
end

function is_gpu_storage_promotion_error(err)
    msg = sprint(showerror, err)
    msg_l = lowercase(msg)
    return occursin("storage types", msg_l) && occursin("cannot be promoted to a concrete type", msg_l)
end

function apply_minimal_regularization!(r)
    params = r.reconParameters[:iterativeReconParams]
    reg = params[:Regularization]
    reg[:TV_spatial] = 0.0
    reg[:TV_spatialTemporal] = 0.0
    if haskey(reg, :TV_temporal)
        reg[:TV_temporal] = 0.0
    end
    reg[:LLR] = 0.0
    params[:iterations] = min(params[:iterations], 3)
    params[:iterationsCG] = min(params[:iterationsCG], 3)
end

function apply_cpu_safe_solver!(r)
    r.reconParameters[:cuda] = false
    r.reconParameters[:cudaSolver] = false
    r.reconParameters[:iterativeReconParams][:subspaceRecon] = false
end

function retry_on_cpu!(r; message::AbstractString, err=nothing)
    if err === nothing
        @warn message
    else
        @warn message exception=(err, catch_backtrace())
    end
    retry_r = deepcopy(r)
    apply_cpu_safe_solver!(retry_r)
    refresh_recon_state!(retry_r)
    return iterativeRecon(retry_r)
end

function run_recon_with_fallback!(r)
    try
        return iterativeRecon(r)
    catch err
        if is_gpu_storage_promotion_error(err) && get(r.reconParameters, :cuda, false) && get(r.reconParameters, :cudaSolver, false)
            @warn "GPU solver type-promotion error detected; retrying on GPU with minimal regularization." exception=(err, catch_backtrace())
            retry_r = deepcopy(r)
            apply_minimal_regularization!(retry_r)
            try
                return iterativeRecon(retry_r)
            catch err2
                return retry_on_cpu!(r; message="Minimal-regularization GPU retry failed; retrying with CPU solver on a fresh recon copy.", err=err2)
            end
        end

        if is_gpu_oom(err) && get(r.reconParameters, :cuda, false)
            target_channels = oom_retry_channel_limit(r)
            if target_channels < size(r.data.kdata, 6)
                @warn "GPU OOM detected; retrying iterative reconstruction on GPU with fewer channels." exception=(err, catch_backtrace()) target_channels=target_channels
                retry_r = deepcopy(r)
                apply_channel_limit!(retry_r, target_channels; reason="GPU OOM retry")
                GC.gc()
                if @isdefined(CUDA)
                    try
                        CUDA.reclaim()
                    catch reclaim_err
                        @warn "CUDA.reclaim() failed during OOM recovery; continuing with reduced-channel GPU retry" exception=(reclaim_err, catch_backtrace())
                    end
                end
                try
                    return iterativeRecon(retry_r)
                catch retry_err
                    if is_gpu_storage_promotion_error(retry_err) || is_gpu_oom(retry_err)
                        return retry_on_cpu!(retry_r; message="Reduced-channel GPU retry failed; retrying with CPU solver on the reduced state.", err=retry_err)
                    end
                    rethrow(retry_err)
                end
            end

            @warn "GPU OOM detected but no smaller channel retry is available; retrying iterative reconstruction in CPU mode." exception=(err, catch_backtrace())
            GC.gc()
            if @isdefined(CUDA)
                try
                    CUDA.reclaim()
                catch reclaim_err
                    @warn "CUDA.reclaim() failed during OOM recovery; continuing with CPU retry" exception=(reclaim_err, catch_backtrace())
                end
            end
            return retry_on_cpu!(r; message="Retrying iterative reconstruction in CPU mode after GPU OOM.")
        end
        rethrow(err)
    end
end

println("=== iterativeRecon ===")
flush(stdout)
r3 = run_recon_with_fallback!(r2)

println("=== post-iterativeRecon inspection ===")
inspect("r3.imgData.signal", r3.imgData.signal)
inspect_reconparams(r3, [:subspaceBasis, :sensMaps])
println("r3.scanParameters keys: ", collect(keys(r3.scanParameters)))
println("Variables r, r2, r3 remain bound in Main for further interactive inspection.")
println("Masking/upsampling were intentionally NOT run in this debug script.")
flush(stdout)

compare_dir = "data/phantom/compare_debug"
isdir(compare_dir) || mkpath(compare_dir)
out_h5 = saveasImDataParams(r3, path=compare_dir, name="debugcompare")
println("Wrote debug h5 (pre-mask/upsample): ", out_h5)
flush(stdout)
