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

    # CUDA_ERROR_NOT_INITIALIZED right after a GPU is released by a prior job
    # on the same node is transient (driver re-init race), not a permanent
    # unavailability -- retry briefly before falling back to CPU, since this
    # codebase's subspace/CASPR reconstruction path has no CPU implementation
    # and will hard-fail later if CUDA_AVAILABLE ends up false.
    max_attempts = 5
    for attempt in 1:max_attempts
        try
            @eval using CUDA
            CUDA.device!(0) # Set the GPU device to use when explicitly requested
            @info "CUDA initialized for reconstruction" device=CUDA.name(CUDA.device()) attempt=attempt
            return true
        catch err
            if attempt == max_attempts
                @warn "CUDA initialization failed after $(max_attempts) attempts; continuing in CPU-only mode" exception=(err, catch_backtrace())
                return false
            end
            @warn "CUDA initialization attempt $(attempt)/$(max_attempts) failed; retrying" exception=(err, catch_backtrace())
            sleep(5)
        end
    end
    return false
end

const CUDA_AVAILABLE = init_cuda_available()
println("[BACKEND] RECON_USE_CUDA=$(RECON_USE_CUDA) CUDA_AVAILABLE=$(CUDA_AVAILABLE)")

function log_label_summary(r, stage::AbstractString)
    TRACE_SORTDATA || return
    labels = r.data.labels
    acc_idx = Int.(vec(labels[:LabelLookupTable][1]))
    center_mask = (labels[:kz][acc_idx] .== 0) .&
                  (labels[:ky][acc_idx] .== 0) .&
                  (labels[:echo][acc_idx] .== 0) .&
                  (labels[:dyn][acc_idx] .== 0)
    println("[SORTDATA] $(stage)")
    println("[SORTDATA] n_profiles=$(length(acc_idx))")
    println("[SORTDATA] n_center_profiles=$(sum(center_mask))")
    println("[SORTDATA] unique_ky=$(length(unique(labels[:ky][acc_idx])))")
    println("[SORTDATA] unique_kz=$(length(unique(labels[:kz][acc_idx])))")
    println("[SORTDATA] unique_echo=$(length(unique(labels[:echo][acc_idx])))")
    println("[SORTDATA] unique_dyn=$(length(unique(labels[:dyn][acc_idx])))")
    println("[SORTDATA] unique_chan=$(length(unique(labels[:chan][acc_idx])))")
    println("[SORTDATA] unique_interleave=$(length(unique(labels[:extr1][acc_idx])))")
end

function log_recon_dims(r, stage::AbstractString)
    TRACE_SORTDATA || return
    @info "[SORTDATA] $(stage) signal dims" dims=size(r.imgData.signal)
end

function should_fail_geometry(target_encoding::AbstractVector{<:Integer}, spatial_size::AbstractVector{<:Integer})
    target_f = Float64.(target_encoding)
    spatial_f = Float64.(spatial_size)
    # Downsampling from reconstructed grid to encoding grid is expected after
    # oversampling removal; only fail on non-positive sizes or expansions that
    # exceed the configured tolerance above the current image grid.
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

"""
Radial cosine taper over the (ky,kz) index plane: 1.0 within r_inner of center,
0.0 beyond r_outer, smooth cosine transition between.
"""
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

"""
Compute coil sensitivity maps from pre-sorted kdata using a sum-of-squares approach.
kdata layout: (kx, ky, kz, echoes, dyn, chan, interleaves)
Returns sensMaps of size (kx, ky, kz, chan).

Restricts estimation to the reliably-sampled k-space calibration region (near-100%
coverage at the center, falling off with radius for this variable-density CASPR
trajectory -- confirmed empirically, see experiment/TODO.md 2026-09-14 root-cause
entry). Without this, the raw SOS ratio is dominated by aliasing-driven speckle
noise in the poorly-covered outer k-space shell, producing sensMaps that are pure
noise in exactly the central image region where the true object sits (see
experiment/sensmaps_perchannel.png) -- a broken coil-sensitivity input makes
SENSE-style unfolding of the coherent aliasing impossible regardless of downstream
regularization/dictionary constraints.
"""
function compute_sensmaps_sos(kdata::Array{Complex{T}, 7}) where T<:AbstractFloat
    kx, ky, kz, necho, ndyn, nchan, ninter = size(kdata)
    # Average across echoes, dynamics, interleaves → (kx, ky, kz, chan)
    kdata_avg = dropdims(mean(kdata, dims=(4, 5, 7)), dims=(4, 5, 7))
    # Calibration-region taper: full weight to r_inner, cosine taper to zero by r_outer.
    # Empirically swept (experiment/sensmaps_taper_sweep*.png, TODO.md 2026-09-14):
    # radii as large as (25,50) still show severe speckle (barely tapers anything);
    # only a genuinely small calibration region -- (6,10), like standard GRAPPA/ESPIRiT
    # calibration-line counts -- yields smooth, physically-plausible per-channel maps.
    taper = Float32.(radial_calib_taper(ky, kz, 6.0, 10.0))
    kdata_avg = kdata_avg .* reshape(taper, 1, ky, kz, 1)
    # Centered 3D iFFT to image space per coil (matches FFTOp shift convention)
    imgs = similar(kdata_avg)
    for c in 1:nchan
        kspace_c = kdata_avg[:, :, :, c]
        imgs[:, :, :, c] .= fftshift(ifft(ifftshift(kspace_c, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))
    end
    # Sum-of-squares magnitude for normalization
    sos = sqrt.(sum(abs.(imgs).^2, dims=4) .+ eps(T))
    # Normalize each coil image by SOS to get sensitivity maps
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

    # CASPR shot/TFE-indexed kdata (see build_jld2_from_mrtools_bridge.jl
    # --presort-caspr) has temporal (tfe,shot) axes, not spatial (ky,kz), on
    # kdata dims 2/3. compute_sensmaps_sos assumes spatial axes and would
    # silently corrupt sensMaps/reconSize via a bogus iFFT-over-time if run
    # here. Skip the spatial-shape-mismatch recompute for this data and only
    # sanity-check the channel count.
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
            # sensMaps cannot be recomputed for CASPR-indexed data (see
            # refresh_recon_state!), so subset its channel axis to match
            # kdata's reduced channel count instead of deleting it.
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

filename = length(ARGS) >= 1 ? ARGS[1] : "data/phantom/csbffe_from_mrtools_presort.jld2"
bypass_sort = "--bypass-sort" in ARGS
motionCorrection = false # No soft-gating for phantom scan

# Load raw data with some preprocessing steps performed
r = jldopen(filename)["r"]
r.pathProc = dirname(replace(filename, "exp_raw" => "exp_pro"))
if !isdir(r.pathProc)
    mkpath(r.pathProc)  # This will also create any necessary parent directories
end

has_phase_corr = hasproperty(r.data, :phaseCorrData) && size(r.data.phaseCorrData, 2) > 0
has_noise_cov = haskey(r.scanParameters, :Psi)

# Detect data type: KdataRaw (needs sorting) vs KdataPreprocessed (already sorted)
# --bypass-sort forces skipping sortData() even for KdataRaw
is_preprocessed = isa(r.data, ReconBMRR.KdataPreprocessed) || bypass_sort

@info "Input data type: $(typeof(r.data).name.name)"
@info "Is preprocessed: $is_preprocessed"

# Preprocessing
if has_phase_corr
    if is_preprocessed
        @warn "Phase correction data present but data is already preprocessed; skipping phaseCorrDataBipolar!"
    else
        phaseCorrDataBipolar!(r)
    end
else
    @info "Skipping phaseCorrDataBipolar!: no phase correction data present in input JLD2"
end

# Sort data if it's raw (KdataRaw); if already preprocessed, skip
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
    # Build matching profileOrder: (2, ky, kz, echoes, dyn, interleaves)
    profileOrder = zeros(Int, 2, num_ky, num_kz, num_echoes, num_dyn, num_inter)
    for i in 1:length(acc_idx)
        col = acc_idx[i]
        ki = Int(ky_v[i]) - ky_min + 1
        zi = Int(kz_v[i]) - kz_min + 1
        ei = Int(echo_v[i]) + 1
        di = Int(dyn_v[i]) + 1
        ii = Int(extr1_v[i]) + 1
        # ReconBMRR preprocessing expects positive 1-based ky/kz profile coordinates.
        profileOrder[1, ki, zi, ei, di, ii] = ki
        profileOrder[2, ki, zi, ei, di, ii] = zi
    end
    new_traj = ReconBMRR.Cartesian3D(profileOrder, :Cartesian3D)
    r2 = ReconBMRR.ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters,
                               kdata_pre, new_traj, r.performedMethods, r.imgData)
else
    # Validate that center-profile repetitions exist before sortData()
    log_label_summary(r, "before sortData")
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
    log_label_summary(r2, "after sortData")
    log_recon_dims(r2, "after sortData")
end

# Applies the bipolar phase-correction fit (whether computed by the legacy
# phaseCorrDataBipolar!(r) path above, or precomputed by the bridge builder
# for the --presort/--presort-caspr path) to r2's KdataPreprocessed data.
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

# Cartesian iterative reconstruction expects interleaves folded into dynamics.
needs_interleave_collapse = typeof(r2.traj) == ReconBMRR.Cartesian3D && ndims(r2.data.kdata) == 7 && size(r2.data.kdata, 7) > 1

if needs_interleave_collapse
    @info "Collapsing interleaves into dynamics for Cartesian reconstruction" dyn=size(r2.data.kdata, 5) interleaves=size(r2.data.kdata, 7)
    changeInterleavesToDynamics!(r2)
    log_recon_dims(r2, "after changeInterleavesToDynamics")
else
    @info "Skipping changeInterleavesToDynamics!: data is already preprocessed or bypass-sort was used"
end

# Set regularization and recon parameters
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

# constructOperators() has no CPU implementation for subspaceRecon (asserts
# reconParameters[:cuda] && cudaSolver); fail fast with a clear message here
# instead of a deep AssertionError inside Reconstruction.jl.
if r2.reconParameters[:iterativeReconParams][:subspaceRecon] && !(r2.reconParameters[:cuda] && r2.reconParameters[:cudaSolver])
    error("Subspace/CASPR reconstruction requires a working CUDA device, but CUDA_AVAILABLE=$(CUDA_AVAILABLE) " *
          "(cuda=$(r2.reconParameters[:cuda]), cudaSolver=$(r2.reconParameters[:cudaSolver])). " *
          "Resubmit on a node with a functional GPU; there is no CPU fallback for this path.")
end

@info "Reconstruction backend configuration" requested_cuda=RECON_USE_CUDA cuda=CUDA_AVAILABLE cudaSolver=r2.reconParameters[:cudaSolver] admm_iterations=r2.reconParameters[:iterativeReconParams][:iterations] cg_iterations=r2.reconParameters[:iterativeReconParams][:iterationsCG]
println("[BACKEND] cuda=$(r2.reconParameters[:cuda]) cudaSolver=$(r2.reconParameters[:cudaSolver])")

function parse_expected_shape(value::String)
    parts = split(value, ',')
    dims = Int[]
    for p in parts
        s = strip(p)
        isempty(s) && continue
        push!(dims, parse(Int, s))
    end
    return dims
end

function count_csv_rows(csv_path::String)
    n = 0
    open(csv_path, "r") do io
        first = true
        for _ in eachline(io)
            if first
                first = false
                continue
            end
            n += 1
        end
    end
    return n
end

function validate_export_output!(h5_path::String)
    required_groups = split(get(ENV, "RECON_REQUIRED_GROUPS", "ImDataParams,MotionParams,RelaxParams"), ',')
    required_groups = [strip(g) for g in required_groups if !isempty(strip(g))]

    expected_shape_env = get(ENV, "RECON_EXPECT_SIGNAL_SHAPE", "")
    expected_shape = isempty(expected_shape_env) ? Int[] : parse_expected_shape(expected_shape_env)

    h5open(h5_path, "r") do fid
        groups = Set(String.(collect(keys(fid))))
        missing = [g for g in required_groups if !(g in groups)]
        if !isempty(missing)
            error("Export validation failed: missing required groups $(missing) in $(h5_path)")
        end

        if !haskey(fid, "ImDataParams/signal")
            error("Export validation failed: missing ImDataParams/signal in $(h5_path)")
        end

        sig_shape = Int[size(fid["ImDataParams/signal"])...]
        if length(sig_shape) != 5
            error("Export validation failed: expected rank-5 ImDataParams/signal, got shape $(Tuple(sig_shape))")
        end

        if !isempty(expected_shape) && sig_shape != expected_shape
            error("Export validation failed: signal shape $(Tuple(sig_shape)) != expected $(Tuple(expected_shape))")
        end

        println("[VALIDATION] H5 groups and signal shape OK: groups=$(collect(groups)) shape=$(Tuple(sig_shape))")

        csv_path = get(ENV, "RECON_SIGNAL_CHUNK_CSV", "")
        if !isempty(csv_path)
            if !isfile(csv_path)
                error("Export validation failed: RECON_SIGNAL_CHUNK_CSV does not exist: $(csv_path)")
            end
            csv_rows = count_csv_rows(csv_path)
            expected_rows = sig_shape[2] * sig_shape[3] # dyn * z for compare-format signal
            if csv_rows != expected_rows
                error("Export validation failed: CSV rows $(csv_rows) != dyn*z $(expected_rows)")
            end
            println("[VALIDATION] CSV rows OK: rows=$(csv_rows) dyn*z=$(expected_rows)")
        end
    end
end

if motionCorrection
    softGatingWeights!(r2)
end

apply_channel_limit!(r2, REQUESTED_MAX_CHANNELS; reason="preconfigured low-memory mode")

# Readout (kx) oversampling is normally cropped from the image AFTER
# reconstruction (upsampleRecVoxelSize!/removeOversampling! below), which
# means the full oversampled kx grid dominates GPU memory during
# constructOperators()/ADMM. Since KxOversampling exactly explains
# kx=200 -> nominal encodingSize[1]=100 here, crop it from k-space instead
# (ifft -> center-crop -> fft along kx), before building the reconstruction
# operators. This is the same physical crop, just done earlier, and is safe
# because the readout oversampling factor guarantees no aliasing in the
# cropped region.
const RECON_REDUCE_KX_OVERSAMPLING = get(ENV, "RECON_REDUCE_KX_OVERSAMPLING", "1") == "1"

function remove_kx_readout_oversampling!(r)
    target_kx = Int(r.scanParameters[:encodingSize][1])
    kd = r.data.kdata
    cur_kx = size(kd, 1)
    if cur_kx <= target_kx
        @info "No kx readout oversampling to remove" cur_kx target_kx
        return
    end
    @info "Removing kx readout oversampling before reconstruction to reduce GPU memory" cur_kx target_kx
    img = fftshift(ifft(ifftshift(kd, 1), 1), 1)
    lo = div(cur_kx - target_kx, 2) + 1
    hi = lo + target_kx - 1
    img = img[lo:hi, :, :, :, :, :, :]
    kd_new = ifftshift(fft(ifftshift(img, 1), 1), 1)
    r.data.kdata = ComplexF32.(kd_new)
    r.reconParameters[:numKx] = target_kx
    @info "kx readout oversampling removed" new_kdata_size=size(r.data.kdata)
end

if RECON_REDUCE_KX_OVERSAMPLING
    remove_kx_readout_oversampling!(r2)
end

subspaceBasis!(r2)
refresh_recon_state!(r2)

# Perform reconstruction with an automatic low-memory fallback.
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
    # Disabling subspaceRecon silently changes the exported signal's feature
    # dimension from subspaceComponents to raw dynamics, which later fails
    # (or worse, mismatches) in postprocessing's dictionary matching. There is
    # no CPU implementation of the subspace/CASPR path, so a CPU fallback here
    # is not a true equivalent retry -- fail loudly instead of producing an
    # H5 with an incompatible feature axis.
    if r.reconParameters[:iterativeReconParams][:subspaceRecon]
        error("GPU reconstruction failed and CPU fallback would silently disable subspaceRecon, " *
              "producing a signal with raw-dynamics feature dimension instead of subspaceComponents. " *
              "Resubmit on a GPU with enough VRAM instead of falling back to CPU for this path.")
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

# Perform reconstruction
r3 = run_recon_with_fallback!(r2)

# Apply mask using recomputed sensMaps. NOTE (2026-09-15): an exact `== 0.0`
# check never triggers with real floating-point IFFT-derived sensMaps (confirmed
# empirically: 0 of ~2M voxels are ever exactly zero) -- this was a silent no-op,
# leaving background (no true signal) regions to show raw reconstruction noise
# instead of being zeroed. Use a relative-magnitude threshold instead.
sense_mask_threshold = try
    parse(Float64, get(ENV, "RECON_SENSE_MASK_THRESHOLD", "0.05"))
catch
    0.05
end
sensMagSum = sum(abs.(r3.reconParameters[:sensMaps]), dims=4)
senseMask = sensMagSum[:,:,:,1] .< (sense_mask_threshold * maximum(sensMagSum))
for s = eachslice(r3.imgData.signal, dims=(4, 5, 6, 7))
    @show size(s)
    s[senseMask] .= 0.0
end

# upsampleRecVoxelSize! and removeOversampling! are only valid for 3D data.
# Skip for 2D scans (kz == 1) to avoid out-of-bounds crop errors.
# Guard against invalid or zero voxel sizes, which can produce Inf when computing
# FOV ./ RecVoxelSize in upsampleRecVoxelSize!.
if size(r3.imgData.signal, 3) > 1
    rec_voxel_source = haskey(r3.scanParameters, :RecVoxelSize) ? r3.scanParameters[:RecVoxelSize] : r3.scanParameters[:AcqVoxelSize]
    rec_voxel = copy(rec_voxel_source)
    spatial_size = Int[size(r3.imgData.signal, 1), size(r3.imgData.signal, 2), size(r3.imgData.signal, 3)]
    fov = haskey(r3.scanParameters, :FOV) ? Float32.(vec(r3.scanParameters[:FOV])) : Float32[]

    # Prefer the reconstruction voxel size that matches template geometry. Fall back to
    # AcqVoxelSize only when RecVoxelSize is missing or unusable.
    if any(.!(isfinite.(rec_voxel) .& (rec_voxel .> 0)))
        fallback_voxel = haskey(r3.scanParameters, :AcqVoxelSize) ? copy(r3.scanParameters[:AcqVoxelSize]) : rec_voxel
        if all(isfinite.(fallback_voxel) .& (fallback_voxel .> 0))
            @warn "RecVoxelSize is invalid after reconstruction; falling back to AcqVoxelSize for geometry normalization." rec_voxel=rec_voxel acq_voxel=fallback_voxel
            rec_voxel = fallback_voxel
        else
            msg = "Detected invalid RecVoxelSize and AcqVoxelSize values: rec=$(rec_voxel), acq=$(fallback_voxel). Cannot safely run geometry normalization."
            if STRICT_ENCODING_MATCH
                error(msg * " RECON_STRICT_ENCODING_MATCH=1, failing reconstruction.")
            end
            @warn msg * " Skipping voxel upsampling to avoid Inf in FOV/RecVoxelSize."
        end
    elseif length(fov) < 3
        msg = "Missing/invalid FOV metadata; reconstructed grid is $(spatial_size)."
        if STRICT_ENCODING_MATCH
            error(msg * " RECON_STRICT_ENCODING_MATCH=1, failing reconstruction.")
        end
        @warn msg * " Skipping upsample/removeOversampling and preserving reconstructed grid."
        r3.scanParameters[:encodingSize] = Int32.(spatial_size)
        r3.scanParameters[:FOV] = Float32.(spatial_size) .* Float32.(rec_voxel)
    else
        target_encoding = round.(Int, fov[1:3] ./ Float32.(rec_voxel))
        # Guard against placeholder FOV (e.g. [1,1,1]) that would collapse output to 1x1x1,
        # and against impossible crops larger than the current reconstructed image.
        if should_fail_geometry(target_encoding, spatial_size)
            msg = "Inconsistent FOV/voxel metadata gives encodingSize=$(target_encoding) for image size=$(spatial_size)."
            if STRICT_ENCODING_MATCH
                error(msg * " RECON_STRICT_ENCODING_MATCH=1, failing reconstruction.")
            end
            @warn msg * " Skipping upsample/removeOversampling and preserving reconstructed grid."
            r3.scanParameters[:encodingSize] = Int32.(spatial_size)
            r3.scanParameters[:FOV] = Float32.(spatial_size) .* Float32.(rec_voxel)
        else
            r3.scanParameters[:RecVoxelSize] = rec_voxel
            upsampleRecVoxelSize!(r3)
            removeOversampling!(r3)
        end
    end
else
    @info "2D scan detected (kz=1): skipping upsampleRecVoxelSize! and removeOversampling!"
end
out_h5 = saveasImDataParams(r3, name="subspace")
println("[VALIDATION] Export file: $(out_h5)")
validate_export_output!(out_h5)
