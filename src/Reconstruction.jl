export iterativeRecon

  
mutable struct initializedVariables{T}
    data::Array{Complex{T}, 7}
    traj::Union{Array{T, 5}, Nothing}
    reconSize::Tuple
    num_kx::Int
    num_chan::Int
    num_echoes::Int
    num_kz::Int
    num_ky::Int
    num_dyn::Int
    num_components::Union{Int, Nothing}
    num_dyn_virt_echoes::Int
    numMotionStates::Int
    numMotionStatesDeltaB0::Union{Int, Nothing}
    numMotionStatesRegulizer::Int
    bCurve::Vector{T}
end

function initializeVariables(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false) where T<:AbstractFloat
    data = r.data.kdata
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    if typeof(r.traj) == Cartesian3D
        traj = nothing
        reconSize = size(r.data.kdata)[1:3]
    else
        traj = r.traj.kdataNodes
        reconSize = deepcopy(r.scanParameters[:encodingSize]) # Init with encodingSize
    end

    num_kx, num_chan, num_echoes, num_kz, num_ky, num_dyn = numKx(r), numChan(r), numEchoes(r), numKz(r), numKy(r), numDyn(r)

    ## get reconSize
    if coilWise
        if r.traj.name == :CartesianSpiral
            if :sensMaps in keys(r.reconParameters)
                reconSize = size(r.reconParameters[:sensMaps])[1:3] 
            else
                reconSize = (size(data,1), r.reconParameters[:numKy], r.reconParameters[:numKz])
            end
        end
    else
        reconSize = size(r.reconParameters[:sensMaps])[1:3]
    end
    reconSize = Tuple(reconSize)

    if params[:subspaceRecon]
        num_components = params[:subspaceComponents]
        num_dyn_virt_echoes = num_components
    else
        num_components = nothing
        num_dyn_virt_echoes = num_dyn
    end

    numMotionStates, bCurve = getBcurve(r)
    numMotionStatesDeltaB0 = nothing
    numMotionStatesRegulizer = numMotionStates

    return initializedVariables(data, traj, reconSize, num_kx, num_chan, num_echoes, num_kz, 
        num_ky, num_dyn, num_components, num_dyn_virt_echoes, numMotionStates, numMotionStatesDeltaB0, 
        numMotionStatesRegulizer, T.(bCurve))
end


function constructOperators(r::ReconParams{<:Union{KdataPreprocessed{T}, KdataRecon{T}}, <:AbstractTrajectory}, 
    weightsMasked::Array, numContr::Int, reconSize::Tuple, coilWise::Bool, num_chan::Int; 
    profiles::Array=[], trajTemp::Array=[], num_components=nothing, numMotionStatesDeltaB0=nothing, echoTime=nothing, csm=nothing) where T<:AbstractFloat
    # if nan values occur, the kernelSize or oversamplingFactor of the nufft can be changed.

    cudaSolver = r.reconParameters[:cudaSolver] # Check if all operators are CUDA compatible and optimizer can run on the GPU
    if !r.reconParameters[:cuda] && cudaSolver
        error("CUDA solver is enabled but CUDA is not enabled in reconParameters.")
    end

    ## FFT operator
    if typeof(r.traj) == Cartesian3D
        if cudaSolver
            @debug("FFTOp on GPU")
            ft = [LinearOperatorCollection.FFTOp(Complex{T}; shape=reconSize, unitary=false, S=CuArray{Complex{T}, 1, CUDA.Mem.DeviceBuffer}) for j=1:numContr]
        else            
            @debug("cuFFTOp (not fully CUDA supported)")
            ft = [ReconBMRR.cuFFTOp(Complex{T}, reconSize; cuda=r.reconParameters[:cuda], unitary=false) for j=1:numContr]
        end
    end

    ## Subspace recon options
    if num_components !== nothing #&& 1 == 0
        num_echoes = ReconBMRR.numEchoes(r)
        if typeof(r.traj) != Cartesian3D
            @warn("Expect CASPR trajectory for subspace reconstruction.")
        end
        @assert r.reconParameters[:cuda] && cudaSolver
        basis = r.reconParameters[:subspaceBasis]
        subOp = CuCasprSubspaceOp(profiles, basis, reconSize, numDyn(r), num_chan, weights)
        @debug("CasprSubspaceOp on GPU")
        @debug("WeightingOp on GPU")
        numContr = num_echoes*num_components
        cuWeightsMasked = CuArray(weightsMasked)
        E = CuCompositeOp(LinearOperatorCollection.WeightingOp(cuWeightsMasked), subOp, isWeighting=true)
        E = CuCompositeOp(E, CuDiagOp(repeat([ft[1]], outer=num_chan*numContr)...))
    else
        if !r.reconParameters[:kspOrdered] && typeof(r.traj) == Cartesian3D
            if r.reconParameters[:cuda] && cudaSolver
                @debug("WeightingOp on GPU")
                @debug("CasprOp on GPU")
                cuWeightsMasked = CuArray(weightsMasked)

                if get(r.reconParameters[:iterativeReconParams], :diffReconTFE, false)
                    E = CuCompositeOp(LinearOperatorCollection.WeightingOp(cuWeightsMasked), 
                        CuCasprOpDiff(profiles, reconSize, numContr, num_chan, r.scanParameters[:FlipAngle], r.scanParameters[:TR]), isWeighting=true)
                else
                    E = CuCompositeOp(LinearOperatorCollection.WeightingOp(cuWeightsMasked), 
                        CuCasprOp(profiles, reconSize, numContr, num_chan), isWeighting=true)
                end
                E = CuCompositeOp(E, CuDiagOp(repeat(ft, outer=num_chan)...))
            else
                @debug("WeightingOp on CPU")
                @debug("CasprOp on CPU")
                E = CompositeOp(LinearOperatorCollection.WeightingOp(weightsMasked), 
                    CasprOp(profiles, reconSize, numContr, num_chan))
                E = CompositeOp(E, DiagOp(repeat(ft, outer=num_chan)...))
            end
        else
            if r.reconParameters[:cuda] && cudaSolver
                @debug("WeightingOp on GPU")
                cuWeightsMasked = CuArray(weightsMasked)
                weighting_op = LinearOperatorCollection.WeightingOp(cuWeightsMasked)
                E = CuCompositeOp(weighting_op, CuDiagOp(repeat(ft, outer=num_chan)...))
            else                    
                @debug("WeightingOp on CPU")
                # E = MRIOperators.CompositeOp(LinearOperatorCollection.WeightingOp(Complex{T}, weights=weightsMasked), DiagOp(ft...), isWeighting=true)
                E = CompositeOp(LinearOperatorCollection.WeightingOp(weightsMasked),
                    DiagOp(repeat(ft, outer=num_chan)...))
            end
        end
    end

    # Sensitivity operator
    if coilWise
        Efull = E
    else
        if csm === nothing
            csm = rescaleSensMaps(r.reconParameters[:sensMaps])
        else
            csm = rescaleSensMaps(csm)
        end
        if r.reconParameters[:cuda] && cudaSolver
            @debug("SensitivityOp on GPU")
            Efull = CuCompositeOp(E, CuSensitivityOp2(reshape(csm, prod(reconSize), num_chan), numContr))
        else
            @debug("SensitivityOp on CPU")
            Efull = CompositeOp(E, SensitivityOp2(reshape(csm, prod(reconSize), num_chan), numContr))
        end
    end
    return Efull
end

function getRegularization(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}, reconSize::Tuple, 
    numEchoesNumDyn::Int, numMotionStates::Int, numCoils::Int) where T<:AbstractFloat
    if r.reconParameters[:cudaSolver]
        opType = CuArray{Complex{T}, 1, CUDA.Mem.DeviceBuffer}
    else
        opType = Vector{Complex{T}}
    end

    if reconSize[3] == 1
        spatialDims = [1,2]
    else
        spatialDims = [1,2,3]
    end

    reg = Vector{AbstractRegularization}()
    regTrafo = []
    if r.reconParameters[:iterativeReconParams][:Regularization][:L1Wavelet_spatial] > 0
        error("L1Wavelet_spatial not implemented.")
    end
    if r.reconParameters[:iterativeReconParams][:Regularization][:LLR] > 0
        push!(reg, LLRRegularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:LLR]), 
            shape=(reconSize[1], reconSize[2], reconSize[3], numEchoesNumDyn*numMotionStates*numCoils), 
            blockSize=(10,10,10, numEchoesNumDyn*numMotionStates*numCoils)))
            # blockSize=(64,64,64, numEchoesNumDyn*numMotionStates*numCoils), fullyOverlapping=true))
            # blockSize=(2,2,2, numEchoesNumDyn*numMotionStates*numCoils)))
        push!(regTrafo, opEye(T, prod((reconSize[1], reconSize[2], reconSize[3], 
            numEchoesNumDyn*numMotionStates*numCoils)), S=opType))
    end
    if r.reconParameters[:iterativeReconParams][:solver] == RegularizedLeastSquares.ADMM 
        if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal] > 0 && numMotionStates > 1
            push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatialTemporal])))
            push!(regTrafo, LinearOperatorCollection.GradientOp(T; shape=(reconSize..., 
                numEchoesNumDyn, numMotionStates, numCoils), dims=[1,2,3,5], S=opType))
        else
            if r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial] > 0
                if :subspaceWeights in keys(r.reconParameters)
                    push!(reg, L1RegularizationDyn(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial]), r.reconParameters[:subspaceWeights], 
                        reconSize; numDyn=numEchoesNumDyn*numMotionStates*numCoils))
                else
                    push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_spatial])))
                end
                push!(regTrafo, LinearOperatorCollection.GradientOp(T; shape=(reconSize..., 
                    numEchoesNumDyn*numMotionStates*numCoils), dims=[1,2,3], S=opType))
            end
            if r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal] > 0 && numMotionStates > 1
                push!(reg, L1Regularization(T.(r.reconParameters[:iterativeReconParams][:Regularization][:TV_temporal])))
                push!(regTrafo, LinearOperatorCollection.GradientOp(T; shape=(reconSize..., 
                    numEchoesNumDyn, numMotionStates, numCoils), dims=5, S=opType))
            end
        end
    else
        error("Regularization for chosen solver not implemented.")
    end
    return reg, regTrafo
end

function iterativeRecon(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}; coilWise::Bool=false, init=nothing) where T<:AbstractFloat    
    @info("Iterative reconstruction.")
    params = r.reconParameters[:iterativeReconParams] # Recon paramters
    var = initializeVariables(r, coilWise=coilWise)
    
    numContr = var.num_echoes * var.num_dyn * var.numMotionStates 

    # Get regularization
    if coilWise
        reg, regTrafos = getRegularization(r, var.reconSize, var.num_echoes*var.num_dyn_virt_echoes, 
            var.numMotionStatesRegulizer, var.num_chan)
    else
        reg, regTrafos = getRegularization(r, var.reconSize, var.num_echoes*var.num_dyn_virt_echoes, 
            var.numMotionStatesRegulizer, 1)
    end

    @debug("Do motion sorting and get sampling mask.")
    # Preprocessing maybe outsource later
    if !r.reconParameters[:kspOrdered]
        dataTemp = permutedims(var.data, [1, 2, 3, 4, 5, 7, 6])
        dataTemp = reshape(dataTemp,:)
        num_samples = r.reconParameters[:sdcCartesian] #reshape(num_samples,:)
    else
        dataTemp = reshape(permutedims(var.data, [1, 2, 3, 4, 5, 7, 6]), :)
    end
    
    # numSamplesTemp = reshape(permutedims(r.reconParameters[:num_samples], [1, 2, 3, 4, 5, 7, 8, 6]), :)
    samplingMask = (dataTemp .== 0)

    if !r.reconParameters[:kspOrdered]
        weightsMasked = Complex{T}.(ones(Complex{T}, length(dataTemp)) ./ sqrt.(prod(var.reconSize) * reshape(num_samples,:)))
    else
        weightsMasked = Complex{T}.(ones(Complex{T}, length(dataTemp)) ./ sqrt(prod(var.reconSize)))
    end
    weightsMasked[isnan.(weightsMasked)] .= 0
    weightsMasked[samplingMask] .= 0
    if haskey(r.reconParameters, :softGatingWeights) && var.numMotionStates == 1 
        @info("Add softgating weights")
        weightsMasked .*= reshape(r.reconParameters[:softGatingWeights],:)
    end
    dataTemp .= dataTemp .* weightsMasked
    if !r.reconParameters[:kspOrdered]
        profiles = r.traj.profileOrder
        profiles = reshape(profiles, 2, size(profiles,2), size(profiles,3), :)
    else
        profiles = []
    end
    Efull = constructOperators(r, weightsMasked, numContr, var.reconSize, coilWise, var.num_chan, 
        profiles=profiles, num_components=var.num_components)

    @debug("Create linear solver.")
    if init !== nothing
        startVector = reshape(init,:)
    else
        startVector = Vector{Complex{T}}(undef, 0)
    end

    # Extract only the parameters that RegularizedLeastSquares.ADMM solver accepts
    solverParams = Dict{Symbol, Any}()
    for key in keys(params)
        if key in [:iterations, :iterationsCG, :vary_rho, :rho, :normalizeReg]
            solverParams[key] = params[key]
        end
    end
    
    solver = createLinearSolver(params[:solver], Efull; reg=reg, regTrafo=regTrafos, solverParams...)

    if params[:verboseIteration]
        @warn("Verbose flag currently not supported.")
        # solver.verbose = true
    end
    if r.reconParameters[:cudaSolver] == true
        dataTemp_gpu = CuArray(dataTemp)
        startVector_gpu = CuArray(startVector)
    else
        dataTemp_gpu = dataTemp
        startVector_gpu = startVector
    end

    solver.verbose = true
    @debug("Perform optimization problem.")
    if init !== nothing
        @debug("Using initial guess.")
        img_gpu = solve!(solver, dataTemp_gpu, startVector=startVector_gpu) 
    else
        img_gpu = solve!(solver, dataTemp_gpu) 
    end
    img = Array(img_gpu)
    img = reshape(img, var.reconSize[1], var.reconSize[2], var.reconSize[3], var.num_echoes, 
        var.num_dyn_virt_echoes, var.numMotionStatesRegulizer, :)
    append!(r.performedMethods, [nameof(var"#self#")])

    # return data
    return_recon = ReconParams(r.filename, r.pathProc, r.scanParameters, r.reconParameters, 
    KdataRecon(dataTemp), r.traj, r.performedMethods, ImgData(img, nothing, nothing, nothing, nothing, nothing, nothing))
    # return_recon.reconParameters[:Efull] = Efull
    return_recon.reconParameters[:weightsMasked] = weightsMasked
    return_recon.reconParameters[:reconSize] = var.reconSize
    return_recon.reconParameters[:var] = var
    return return_recon
end

function rescaleSensMaps(csm::Array{Complex{T}, 4}) where T<:AbstractFloat
    # Compute the sum of squares of magnitudes across coils
    sos = sqrt.(sum(abs.(csm).^2, dims=4))

    # Magnitude normalization
    normalized_array = similar(csm)
    normalized_array .= csm ./ (sos .+ eps())
    normalized_array[abs.(csm) .== 0] .= 0
    return normalized_array
end