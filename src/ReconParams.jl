export ReconParams, ifft_wshift!, fft_wshift!

"""
kspace & reconstructed data types
"""
abstract type AbstractReconData end
abstract type AbstractKdata <: AbstractReconData end
abstract type AbstractTrajectory end
abstract type AbstractNonCartesian <: AbstractTrajectory end
abstract type AbstractCartesian <: AbstractTrajectory end

struct NonCartesian3D <: AbstractNonCartesian
   kdataNodes::Array{<:AbstractFloat}
   name::Symbol
end

struct Cartesian3D <: AbstractCartesian
    profileOrder::Array{Int}
    name::Symbol
end

mutable struct KdataRaw{T<:AbstractFloat} <: AbstractKdata
    accImagData::Matrix{Complex{T}}
    rejImagData::Matrix{Complex{T}}
    phaseCorrData::Matrix{Complex{T}}
    freqCorrData::Matrix{Complex{T}}
    noiseData::Matrix{Complex{T}}
    labels::Dict{Symbol,Any}
end

mutable struct KdataPreprocessed{T<:AbstractFloat} <: AbstractKdata
    kdata::Array{Complex{T},7} # kx, ky, kz, echoes, dynamics, channels, interleaves
    #, motion_state_cart
    ## TODO: channel dimension should be 4th dim
end

mutable struct KdataRecon{T<:AbstractFloat} <: AbstractKdata
    kdata::Union{Vector{Complex{T}}, Array{Complex{T}}}
end

mutable struct ImgData{T<:AbstractFloat} <: AbstractReconData
    signal::Array{Complex{T}}
    water_map::Union{Array{Complex{T}}, Nothing}
    fat_map::Union{Array{Complex{T}}, Nothing}
    r2_star_map::Union{Array{T}, Nothing}
    b0_map::Union{Array{T}, Nothing}
    phasor_map::Union{Array{Complex{T}}, Nothing}
    mask_pygandalf::Union{Array{Bool}, Nothing}
end

"""
Struct describing MRI acquisition

And another environment variable if figures shouldn't be opened in the GUI
    -> GKSwstype="nul"
"""
mutable struct ReconParams{T<:AbstractKdata, V<:AbstractTrajectory, U<:Union{ImgData, Nothing}}
    filename::String
    pathProc::String 
    scanParameters::Dict{Symbol,Any}
    reconParameters::Dict{Symbol,Any}
    data::T
    traj::V
    performedMethods::Vector{Symbol}
    imgData::U
end

function getFilenames(filename::String, pathProc::String)
    splittedFilename = split(filename, ".")
    ## Load raw kspace data from MRecon
    filenameRaw = join(splittedFilename[1:end-1], ".")
    filenameMat = join(split(joinpath(pathProc, basename(filename)), ".")[1:end-1], ".")
    return filenameRaw, filenameMat
end

function setDefaultReconParameters!(r::ReconParams)
    r.reconParameters = Dict(
        :cuda => has_cuda_gpu(),
        :cudaSolver => has_cuda_gpu(),
        :useDoublePrecision => false,
        :artificalUndersampling => 0,
        :export => Dict{Symbol,Any}(),
        :iterativeReconParams => setIterativeReconParams(),
        :motionGating => r.scanParameters[:AcqMode] == "Radial" ? :motionStatesSelfGating! : false,
        :motionGatingParams => setMotionGatingParams(r.scanParameters[:AcqMode] == "Radial"),
        :motionStatesRecon => r.scanParameters[:AcqMode] == "Radial" ? "2" : nothing,
        :noisePreWhitening => false,
        :upsampleRecVoxelSize => r.scanParameters[:AcqMode] == "Radial" ? true : nothing,
        :removeOversampling => true,
        :kspOrdered => false # if k-space gridded
    )
end

function setIterativeReconParams(solver::String="ADMM")
    if solver == "ADMM"
        return Dict(
            :Regularization => Dict(
                :L1Wavelet_spatial => 0.0,
                :LLR => 0.0,
                :TV_spatialTemporal => 0.5,
                :TV_spatial => 0.1,
                :TV_temporal => 0.0, 
            ),
            :solver => RegularizedLeastSquares.ADMM,
            :normalizeReg => RegularizedLeastSquares.MeasurementBasedNormalization(),
            :iterations => 10,
            :iterationsCG => 5,
            :vary_rho => :none,
            :verboseIteration => false, 
            :subspaceRecon => false,
            :subspaceComponents => 5, 
            :spatialDeltaB0 => false
        )
    end
end

function accImagDataLabels(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return [(Int.(r.data.labels[:LabelLookupTable][1]))...]
end

function phaseCorrDataLabels(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return [(Int.(r.data.labels[:LabelLookupTable][3]))...]
end

function numKx(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.accImagData, 1)
end

function numKx(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 1)
end

function numKy(r::ReconParams{KdataRaw{T}, <:AbstractCartesian}) where T<:AbstractFloat
    return  Int(maximum(r.scanParameters[:KyRange]) - minimum(r.scanParameters[:KyRange]) + 1)end

function numKy(r::ReconParams{KdataRaw{T}, <:AbstractNonCartesian}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return  length(unique(r.data.labels[:ky][indices]))
end

function numKy(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 2)
end

function numKz(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return Int(maximum(r.scanParameters[:KzRange]) - minimum(r.scanParameters[:KzRange]) + 1)
end

function numKz(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 3)
end

function numEchoes(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:echo][indices]))
end

function numEchoes(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 4)
end

function numDyn(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:dyn][indices]))
end

function numDyn(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 5)
end

function numChan(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:chan][indices]))
end

function numChan(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 6)
end

function numInterleaves(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat 
    indices = accImagDataLabels(r)
    return length(unique(r.data.labels[:extr1][indices]))
end

function numInterleaves(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    return size(r.data.kdata, 7)
end

function numMotionStates(r::ReconParams{KdataPreprocessed{T}, <:Cartesian3D}) where T<:AbstractFloat
    return size(r.data.kdata, 8)
end

function getBcurve(r::ReconParams{<:AbstractKdata, <:AbstractTrajectory})
    if r.reconParameters[:motionGating] != false
        if r.reconParameters[:motionStatesRecon] == "all"
            numMotionStates = r.reconParameters[:motionGatingParams][:numClusters]
        else
            numMotionStates = parse(Int64, r.reconParameters[:motionStatesRecon])
        end
        bCurve = r.reconParameters[:motionStates]
    else
        numMotionStates = 1
        bCurve = ones(numKy(r)*numDyn(r))
    end
    return numMotionStates, bCurve
end

function ifft_wshift!(data::Array, dim)
    shape = size(data)
    tmpVec = zeros(eltype(data), shape)
    iplan_z = plan_ifft!(tmpVec, dim)
    ReconBMRR.fft_multiply_shift!(iplan_z, data, tmpVec)
end

function fft_wshift!(data::Array, dim)
    shape = size(data)
    tmpVec = zeros(eltype(data), shape)
    plan_z = plan_fft!(tmpVec, dim)
    ReconBMRR.fft_multiply_shift!(plan_z, data, tmpVec)
end

function fft_multiply_shift!(plan::AbstractFFTs.Plan, y::AbstractArray, tmpVec::AbstractArray)
    ifftshift!(tmpVec, y)
    plan * tmpVec
    fftshift!(y, tmpVec)
end