module ReconBMRR

using RegularizedLeastSquares
using Statistics
using MultivariateStats
using LinearAlgebra
using LinearOperatorCollection
using FourierTools
using Distances
using Clustering
using ImageTransformations
using Interpolations
using CUDA
using FLoops
using StatsBase
using HDF5
using ProgressMeter
using DataStructures
using NaNStatistics
using StaticArrays
using NPZ
using Suppressor
using Mmap
using ImageFiltering
using SparseArrays
using DSP
using LsqFit
using Images

"""
    concrete_cuvector_type(::Type{T}) where T

Return the concrete `CuArray{T,1,M}` vector type for element type `T`, as
instantiated by the currently loaded CUDA.jl memory backend.

`CuVector{T}` (i.e. `CuArray{T,1}`) alone is NOT a concrete type: the memory
buffer type parameter `M` is left free, so `CuVector{T}` is a `UnionAll`.
When this abstract alias is passed as an operator's declared storage type
(`S=CuVector{T}`), `LinearOperators.jl` throws
`LinearOperatorException("storage types ... cannot be promoted to a concrete
type")` while composing GPU operators (e.g. inside ADMM), because promoting
two non-concrete types does not yield a concrete result. Using this helper
instead ensures the declared storage type always matches the concrete type of
real `CuArray` instances at runtime. See experiment/TODO.md "Track B" for the
full root-cause analysis.
"""
concrete_cuvector_type(::Type{T}) where {T} = typeof(CuArray{T}(undef, 0))

include("ReconParams.jl")
include("Operators/SensitivityOp2.jl")
include("Operators/FFTOp.jl")
include("Operators/DiagOp.jl")
include("Operators/CasprOp.jl")
include("Operators/CasprSubspaceOp.jl")
include("Operators/CompositeOp.jl")
include("Regularization.jl")
include("Preprocessing.jl")
include("Reconstruction.jl")
include("Postprocessing.jl")
include("Export.jl")

end # module