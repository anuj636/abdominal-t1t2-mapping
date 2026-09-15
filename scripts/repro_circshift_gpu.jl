# Minimal, ReconBMRR-independent repro for the CUDA kernel BoundsError seen
# during FFT adjoint computation in Track B GPU debugging (jobs 6922-6924).
#
# Mirrors exactly what LinearOperatorFFTWExt.fft_multiply_shift! does:
#   ifftshift!(tmpVec, reshape(x, shape))
#   plan * tmpVec
#   fftshift!(reshape(res, shape), tmpVec)
#
# for shape=(200,73,139) (the raw k-space matrix size from this dataset,
# kdata_size=(200, 73, 139, ...)), using a bare CuArray (no SubArray/view,
# no CuDiagOp/LinearOperators involved) to isolate whether the BoundsError
# is a generic GPU kernel bug (odd-sized dims / circshift) or something
# introduced by our operator-composition/slicing code.

using CUDA
using FFTW
using FFTW.AbstractFFTs

println("[REPRO] CUDA functional: ", CUDA.functional())
println("[REPRO] device: ", CUDA.name(CUDA.device()))

shape = (200, 73, 139)
println("[REPRO] shape = ", shape, "  (odd dims: ", filter(iseven, shape) |> x -> shape .% 2, ")")

x_cpu = ComplexF32.(randn(ComplexF32, shape))
x = CuArray(x_cpu)
tmpVec = similar(x)
res = similar(vec(x))

println("[REPRO] Step 1: ifftshift! into tmpVec (bare CuArray, no view/reshape-of-subarray)")
try
    AbstractFFTs.ifftshift!(tmpVec, x)
    CUDA.synchronize()
    println("[REPRO] Step 1 OK")
catch e
    println("[REPRO] Step 1 FAILED: ", e)
    rethrow(e)
end

println("[REPRO] Step 2: ifftshift! into tmpVec, src = reshape(vec(x), shape) (mirrors fft_multiply_shift! exactly)")
xvec = vec(x)
try
    AbstractFFTs.ifftshift!(tmpVec, reshape(xvec, shape))
    CUDA.synchronize()
    println("[REPRO] Step 2 OK")
catch e
    println("[REPRO] Step 2 FAILED: ", e)
    rethrow(e)
end

println("[REPRO] Step 3: ifftshift! into tmpVec, src = reshape(view(xvec, 1:length(xvec)), shape) (mirrors CuDiagOp's view(x, xIdx[i]:xIdx[i+1]-1) slicing)")
try
    xview = view(xvec, 1:length(xvec))
    AbstractFFTs.ifftshift!(tmpVec, reshape(xview, shape))
    CUDA.synchronize()
    println("[REPRO] Step 3 OK")
catch e
    println("[REPRO] Step 3 FAILED: ", e)
    rethrow(e)
end

println("[REPRO] Step 4: full fft_multiply_shift!-equivalent round trip with a real cuFFT plan")
try
    plan = plan_fft!(copy(tmpVec))
    AbstractFFTs.ifftshift!(tmpVec, reshape(xvec, shape))
    plan * tmpVec
    AbstractFFTs.fftshift!(reshape(res, shape), tmpVec)
    CUDA.synchronize()
    println("[REPRO] Step 4 OK")
catch e
    println("[REPRO] Step 4 FAILED: ", e)
    rethrow(e)
end

println("[REPRO] ALL STEPS PASSED")
