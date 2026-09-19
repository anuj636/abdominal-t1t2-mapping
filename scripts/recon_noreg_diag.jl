using Pkg
Pkg.activate(".")
using Revise
using ReconBMRR
using CUDA
CUDA.device!(0)
using JLD2, CodecZlib

# Quick diagnostic (bridge data-fidelity investigation): test whether the ~42x
# smaller raw k-space data amplitude vs phantom_3_5mm.jld2 (mean|abs| 1.93 vs
# 80.8, while noise covariance Psi is essentially identical ~415 in both) means
# the fixed absolute TV_spatial=0.001 regularization weight is far too strong
# for our data's scale, causing over-smoothing/artifacts. Tests TV_spatial=0
# on the already-built (FOV-fixed) JLD2, no rebuild needed.

filename = "data/phantom/dixon_caspr_calib74_with_3_5mmV4_raw_fovfix_caspr_subspace.jld2"
r = jldopen(filename)["r"]
r.pathProc = dirname(filename)
if !isdir(r.pathProc)
    mkpath(r.pathProc)
end

r.reconParameters[:cuda] = true
r.reconParameters[:cudaSolver] = true
r.reconParameters[:motionStatesRecon] = 1
r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] = 0.001 / 42
r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] = 0.0
r.reconParameters[:iterativeReconParams][:Regularization][:LLR] = 0.0
r.reconParameters[:iterativeReconParams][:subspaceRecon] = true
r.reconParameters[:iterativeReconParams][:subspaceComponents] = 5
r.reconParameters[:iterativeReconParams][:iterations] = 15
r.reconParameters[:iterativeReconParams][:iterationsCG] = 10
r.reconParameters[:iterativeReconParams][:vary_rho] = :balance
r.reconParameters[:iterativeReconParams][:rho] = 0.01
r.reconParameters[:prepDictPath] = "src/Files/20241022_dict_caspr_lookLocker_with0deg_31B0.h5"

# 16-channel limit, matching the production default
kd = r.data.kdata
target = 16
r.data.kdata = kd[:, :, :, :, :, 1:target, :]
sm = r.reconParameters[:sensMaps]
r.reconParameters[:sensMaps] = sm[:, :, :, 1:target]
ReconBMRR.computeDensityCompensation!(r)
subspaceBasis!(r)

r3 = iterativeRecon(r)

sensMagSum = sum(abs.(r3.reconParameters[:sensMaps]), dims=4)
senseMask = sensMagSum[:,:,:,1] .< (0.05 * maximum(sensMagSum))
for s = eachslice(r3.imgData.signal, dims=(4, 5, 6, 7))
    s[senseMask] .= 0.0
end
r3.scanParameters[:RecVoxelSize] = r3.scanParameters[:AcqVoxelSize]
get!(r3.scanParameters, :isFINO, false)
out_h5 = saveasImDataParams(r3, name="noreg_diag")
println("[DIAG] Export file: $(out_h5)")
