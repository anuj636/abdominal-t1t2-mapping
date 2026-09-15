# Faithful repro #2 for the CUDA kernel BoundsError seen during FFT adjoint
# computation (jobs 6922-6924). Unlike repro_circshift_gpu.jl (which hand-rolled
# the ifftshift!/fftshift!/plan sequence and PASSED), this script uses the
# REAL production code path:
#   LinearOperatorCollection.FFTOp(...)  -- exact same constructor call as
#                                           Reconstruction.jl:81
#   ReconBMRR.CuDiagOp(repeat([ft1], outer=N)...) -- exact same shared-operator
#                                           repetition pattern as
#                                           Reconstruction.jl:102 (subspace path)
#   adjoint(diagop) * x                  -- mirrors cuDiagOpCTProd's mul! call
#
# Goal: determine whether going through the real FFTOp/CuDiagOp/adjoint wrapper
# stack (as opposed to calling ifftshift!/fftshift!/plan directly) reproduces
# the BoundsError, which would implicate the operator-composition/object-sharing
# pattern rather than a generic GPUArrays/circshift bug.

using CUDA
using LinearAlgebra

# Load ReconBMRR as the real package (same project the recon pipeline uses).
# ReconBMRR itself depends on/re-exports LinearOperatorCollection internally.
using ReconBMRR
using ReconBMRR.LinearOperatorCollection

reconSize = (200, 73, 139)
num_chan = 2
numContr = 10   # num_chan * numContr = 20, matching the crashing job's "i=1/20"
T = Float32

println("[REPRO2] CUDA functional: ", CUDA.functional())
println("[REPRO2] reconSize=$reconSize num_chan=$num_chan numContr=$numContr -> total ops = $(num_chan*numContr)")

println("[REPRO2] Constructing FFTOpImpl exactly as Reconstruction.jl:81 does...")
ft = [LinearOperatorCollection.FFTOp(Complex{T}; shape=reconSize, unitary=false,
                                     S=ReconBMRR.concrete_cuvector_type(Complex{T})) for j=1:1]
ft1 = ft[1]
println("[REPRO2] ft1 type = ", typeof(ft1), " nrow=", ft1.nrow, " ncol=", ft1.ncol)

println("[REPRO2] Constructing CuDiagOp(repeat([ft1], outer=num_chan*numContr)...) exactly as Reconstruction.jl:102 does...")
diagop = ReconBMRR.CuDiagOp(repeat([ft1], outer=num_chan*numContr)...)
println("[REPRO2] diagop nrow=", diagop.nrow, " ncol=", diagop.ncol)

n = prod(reconSize)
total = n * num_chan * numContr
x_cpu = ComplexF32.(randn(ComplexF32, total))
x = CuArray(x_cpu)
res = similar(x)

println("[REPRO2] Calling mul!(res, adjoint(diagop), x) -- mirrors cuDiagOpCTProd i=1 call exactly...")
try
    mul!(res, adjoint(diagop), x)
    CUDA.synchronize()
    println("[REPRO2] mul! adjoint(diagop) OK")
catch e
    println("[REPRO2] mul! adjoint(diagop) FAILED: ", e)
    for (i, frame) in enumerate(stacktrace(catch_backtrace()))
        println("  [$i] ", frame)
    end
    rethrow(e)
end

println("[REPRO2] Calling mul!(res2, diagop, x2) forward direction too, for completeness...")
try
    res2 = similar(x)
    mul!(res2, diagop, x)
    CUDA.synchronize()
    println("[REPRO2] mul! diagop (forward) OK")
catch e
    println("[REPRO2] mul! diagop (forward) FAILED: ", e)
    rethrow(e)
end

println("[REPRO2] ALL STEPS PASSED")
