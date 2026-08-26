using Pkg
Pkg.activate(".") # load the environment from Project.toml
using Revise
using ReconBMRR
using CUDA
CUDA.device!(0) # Set the GPU device to use
using JLD2, CodecZlib
using FFTW
using Statistics

"""
Compute coil sensitivity maps from pre-sorted kdata using a sum-of-squares approach.
kdata layout: (kx, ky, kz, echoes, dyn, chan, interleaves)
Returns sensMaps of size (kx, ky, kz, chan).
"""
function compute_sensmaps_sos(kdata::Array{Complex{T}, 7}) where T<:AbstractFloat
    kx, ky, kz, necho, ndyn, nchan, ninter = size(kdata)
    # Average across echoes, dynamics, interleaves → (kx, ky, kz, chan)
    kdata_avg = dropdims(mean(kdata, dims=(4, 5, 7)), dims=(4, 5, 7))
    # iFFT to image space per coil
    imgs = similar(kdata_avg)
    for c in 1:nchan
        imgs[:, :, :, c] .= ifftshift(ifft(ifftshift(kdata_avg[:, :, :, c])))
    end
    # Sum-of-squares magnitude for normalization
    sos = sqrt.(sum(abs.(imgs).^2, dims=4) .+ eps(T))
    # Normalize each coil image by SOS to get sensitivity maps
    sensmaps = imgs ./ sos
    return sensmaps
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
    
    if has_phase_corr
        applyPhaseCorrDataBipolar!(r2)
    else
        @info "Skipping applyPhaseCorrDataBipolar!: no phase correction fit was computed"
    end
end

if has_noise_cov
    noisePreWhitening!(r2)
else
    @info "Skipping noisePreWhitening!: no Psi noise covariance matrix present in input JLD2"
end

# changeInterleavesToDynamics is only needed for KdataRaw that went through sortData
if !is_preprocessed && !bypass_sort
    changeInterleavesToDynamics!(r2)
else
    @info "Skipping changeInterleavesToDynamics!: data is already preprocessed or bypass-sort was used"
end

# Set regularization and recon parameters
r2.reconParameters[:cuda] = false       # Disable GPU (insufficient VRAM)
r2.reconParameters[:cudaSolver] = false # Disable GPU solver
r2.reconParameters[:motionStatesRecon] = 1
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] = 0.001
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] = 0.0
r2.reconParameters[:iterativeReconParams][:Regularization][:LLR] = 0.0
r2.reconParameters[:iterativeReconParams][:subspaceRecon] = false  # Disable subspace (GPU-only)
r2.reconParameters[:iterativeReconParams][:subspaceComponents] = 5
r2.reconParameters[:iterativeReconParams][:iterations] = 15
r2.reconParameters[:iterativeReconParams][:iterationsCG] = 10
r2.reconParameters[:iterativeReconParams][:vary_rho] = :balance
r2.reconParameters[:iterativeReconParams][:rho] = 0.01
r2.reconParameters[:prepDictPath] = "src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5"

if motionCorrection
    softGatingWeights!(r2)
end
subspaceBasis!(r2)
computeDensityCompensation!(r2)

# Check if stored sensMaps are consistent with actual kdata dimensions.
# The mrtools bridge may embed sensMaps from a different scan.
if haskey(r2.reconParameters, :sensMaps)
    sm = r2.reconParameters[:sensMaps]
    kd = r2.data.kdata
    if size(sm)[1:3] != size(kd)[1:3] || size(sm, 4) != size(kd, 6)
        @warn "sensMaps $(size(sm)) do not match kdata (spatial=$(size(kd)[1:3]), chan=$(size(kd,6))). " *
              "Recomputing sensMaps from kdata using sum-of-squares method."
        r2.reconParameters[:sensMaps] = compute_sensmaps_sos(kd)
        @info "New sensMaps size: $(size(r2.reconParameters[:sensMaps]))"
    end
else
    @info "No sensMaps in JLD2. Computing from kdata using sum-of-squares method."
    r2.reconParameters[:sensMaps] = compute_sensmaps_sos(r2.data.kdata)
    @info "New sensMaps size: $(size(r2.reconParameters[:sensMaps]))"
end

# Perform reconstruction
r3 = iterativeRecon(r2)

# Apply mask using recomputed sensMaps
senseMask = sum(abs.(r3.reconParameters[:sensMaps]), dims=4) .== 0.0
senseMask = senseMask[:,:,:,1]
for s = eachslice(r3.imgData.signal, dims=(4, 5, 6, 7))
    @show size(s)
    s[senseMask] .= 0.0
end

# upsampleRecVoxelSize! and removeOversampling! are only valid for 3D data.
# Skip for 2D scans (kz == 1) to avoid out-of-bounds crop errors.
if size(r3.imgData.signal, 3) > 1
    r3.scanParameters[:RecVoxelSize] = r3.scanParameters[:AcqVoxelSize].-1.0
    upsampleRecVoxelSize!(r3)
    removeOversampling!(r3)
else
    @info "2D scan detected (kz=1): skipping upsampleRecVoxelSize! and removeOversampling!"
end
saveasImDataParams(r3, name="subspace")
