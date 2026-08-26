using Pkg
Pkg.activate(".") # load the environment from Project.toml
# Pkg.activate("/home/jstelter/.julia/environments/cuReconBMRRv2")
using Revise # useful package if something needs to be changed in the Julia code without restarting the Julia session
using ReconBMRR # includes recon code
# using CUDA
# CUDA.device!(0) # Set the GPU device to use
using JLD2, CodecZlib

#filename = "data/phantom/phantom_3_5mm.jld2"
filename = "data/phantom/csbffe_from_mrtools_presort.jld2"
motionCorrrection = false # No soft-gating for phantom scan

# Use conservative settings for GPUs with limited VRAM (e.g. 12-16 GB).
# Set to true for full reconstruction, false for quick testing
lowMemoryMode = false
testMode = true  # Set to false for production

# Load raw data with some preprocessing steps performed
r = jldopen(filename)["r"]
r.pathProc = dirname(replace(filename, "exp_raw" => "exp_pro"))
if !isdir(r.pathProc)
    mkpath(r.pathProc)  # This will also create any necessary parent directories
end

# Preprocessing
phaseCorrDataBipolar!(r)
r2 = sortData(r)
applyPhaseCorrDataBipolar!(r2)
noisePreWhitening!(r2)
changeInterleavesToDynamics!(r2)

# Set regularization and recon parameters
r2.reconParameters[:motionStatesRecon] = 1
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] = 0.001
r2.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] = 0.0
r2.reconParameters[:iterativeReconParams][:Regularization][:LLR] = 0.0
r2.reconParameters[:iterativeReconParams][:subspaceRecon] = true
r2.reconParameters[:iterativeReconParams][:subspaceComponents] = 5
r2.reconParameters[:iterativeReconParams][:iterations] = 15
r2.reconParameters[:iterativeReconParams][:iterationsCG] = 10
r2.reconParameters[:iterativeReconParams][:vary_rho] = :balance
r2.reconParameters[:iterativeReconParams][:rho] = 0.01
r2.reconParameters[:prepDictPath] = "src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5" 

if lowMemoryMode
    r2.reconParameters[:iterativeReconParams][:subspaceRecon] = false
    r2.reconParameters[:iterativeReconParams][:subspaceComponents] = 3
    r2.reconParameters[:iterativeReconParams][:iterations] = 1
    r2.reconParameters[:iterativeReconParams][:iterationsCG] = 1
    # Keep GPU enabled but use reduced settings for limited VRAM (12-16 GB).
    r2.reconParameters[:cuda] = false
    r2.reconParameters[:cudaSolver] = false
end

if testMode
    r2.reconParameters[:iterativeReconParams][:subspaceRecon] = false  # Disable subspace to skip expensive SVD
    r2.reconParameters[:iterativeReconParams][:iterations] = 1  # Minimum iterations
    r2.reconParameters[:iterativeReconParams][:iterationsCG] = 1  # Minimum CG iterations
    r2.reconParameters[:cuda] = false  # Disable GPU (not enough memory)
    r2.reconParameters[:cudaSolver] = false  # Use CPU solver
    @info "Test mode: CPU only, 1 ADMM iteration with 1 CG iteration"
end

if motionCorrrection
    softGatingWeights!(r2)
end

if r2.reconParameters[:iterativeReconParams][:subspaceRecon]
    @info("Computing subspace basis...")
    subspaceBasis!(r2)
else
    @info("Subspace reconstruction disabled, skipping SVD computation")
end

computeDensityCompensation!(r2)
if r2.reconParameters[:cuda] && isdefined(ReconBMRR, :CUDA)
    ReconBMRR.CUDA.reclaim()
end

# Perform reconstruction
r3 = iterativeRecon(r2)

# Apply mask
senseMask = sum(abs.(r3.reconParameters[:sensMaps]), dims=4) .== 0.0 # Masking
senseMask = senseMask[:,:,:,1]
for s = eachslice(r3.imgData.signal, dims=(4, 5, 6, 7))
    @show size(s)
    s[senseMask] .= 0.0
end
r3.scanParameters[:RecVoxelSize] = r3.scanParameters[:AcqVoxelSize].-1.0
upsampleRecVoxelSize!(r3)
removeOversampling!(r3)
saveasImDataParams(r3, name="subspace")