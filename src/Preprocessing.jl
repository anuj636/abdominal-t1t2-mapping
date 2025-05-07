export phaseCorrDataBipolar!, applyPhaseCorrDataBipolar!, noisePreWhitening!, changeInterleavesToDynamics!, softGatingWeights!,
    subspaceBasis!, computeDensityCompensation!, sortData

function phaseCorrDataBipolar!(r::ReconParams{KdataRaw{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    labels = r.data.labels
    indices = phaseCorrDataLabels(r)
    
    # Go to image space
    data = deepcopy(r.data.phaseCorrData)
    ifft_wshift!(data, 1)

    numKx = size(data, 1)
    kx = Array(-numKx/2:numKx/2-1) / (numKx/2)
    function model(x, p)
        x1 = x[1:Int(length(x)/2)]
        x2 = x[Int(length(x)/2+1):end]
        x1 = x1 .* exp.(1im*(kx*pi*p[1]))
        x2 = x2 .* exp.(-1im*(kx*pi*p[1]))
        k1 = abs.(fftshift(fft(ifftshift(x1))))
        k2 = abs.(fftshift(fft(ifftshift(x2))))
        dk1 = k1[2:end] .- k1[1:end-1]
        dk2 = k2[2:end] .- k2[1:end-1]
        return dk1 .- dk2
    end

    params = zeros(T, 2, numEchoes(r))
    for echo in 1:numEchoes(r)
        indices_echo = indices[labels[:echo][indices] .== echo-1]
        indices_pos = indices_echo[labels[:sign][indices_echo] .== 1]
        indices_neg = indices_echo[labels[:sign][indices_echo] .== -1]

        # indices_pos = indices_echo[(labels[:sign][indices_echo] .== 1) .& (labels[:card][indices_echo] .== 5)]
        # indices_neg = indices_echo[(labels[:sign][indices_echo] .== -1) .& (labels[:card][indices_echo] .== 5)]

        mask_pos = in.(vec(r.data.labels[:LabelLookupTable][3]), Ref(indices_pos))
        mask_neg = in.(vec(r.data.labels[:LabelLookupTable][3]), Ref(indices_neg))

        data_pos = mean(data[:,mask_pos], dims=2)[:,1]
        data_neg = mean(data[:,mask_neg], dims=2)[:,1]

        # plot(abs.(data_pos))
        # plot!(abs.(data_neg))

        data_fit = vcat(data_neg, data_pos)
        # data_fit = vcat(data_pos, data_neg)
        # @show mean(abs.(model(data_fit, 0.0*data_pos))) / mean(abs.(data_fit)) #model(data_fit, 0.0*data_pos)
        # fit = curve_fit(model, data_fit, 0.0 * data_pos[1:end], [0.0, 0.0])
        fit = curve_fit(model, data_fit, 0.0 * data_pos[1:end-1], [0.0])

        params[1, echo] = 0.0 #fit.param[1]
        params[2, echo] = fit.param[1]
        @show mean(abs.(fit.resid)) / mean(abs.(data_fit))
    end
    r.reconParameters[:phaseCorrDataBipolar] = params
    append!(r.performedMethods, [nameof(var"#self#")])
end

function applyPhaseCorrDataBipolar!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    if haskey(r.reconParameters, :phaseCorrDataBipolar)
        # Correction
        p = r.reconParameters[:phaseCorrDataBipolar]
        numKx = size(r.data.kdata, 1)
        x = Array(-numKx/2:numKx/2-1) / (numKx/2)
        data = r.data.kdata
        ifft_wshift!(data, 1) #ifftshift(ifft(ifftshift(data, (1)), 1), (1))

        for j in 1:Int(numEchoes(r))
            phase = exp.(1im*(-1)^(j)*(x*pi*p[2,j].+p[1,j])) 
            for i in 1:length(phase)
                data[i,:,:,j,:,:,:,:,:] .*= phase[i]
            end
        end
        fft_wshift!(data, 1) 
        append!(r.performedMethods, [nameof(var"#self#")])
    end
end

function noisePreWhitening!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat
    if :Psi in keys(r.scanParameters)
        @debug("Perform noise pre-whitening.")
        data = r.data.kdata
        psi = r.scanParameters[:Psi]
        csm = r.reconParameters[:sensMaps]
        # Compute the Cholesky decomposition
        L_inv = inv(cholesky(psi).L)

        # Define the transformation function to be applied in-place
        function transform_fn!(slice)
            slice .= L_inv * slice
        end

        # Apply the transformation to the data using in-place mapslices with multi-threading
        @floop for s = eachslice(data, dims=(1, 2, 3, 4, 5, 7))
            transform_fn!(s)
        end

        # Similarly, apply the transformation to csm if needed
        @floop for s = eachslice(csm, dims=(1, 2, 3)) 
            transform_fn!(s)
        end

        r.data = KdataPreprocessed(data)
        r.reconParameters[:sensMaps] = csm
        append!(r.performedMethods, [nameof(var"#self#")])
    else
        error("Noise correlation matrix not found. No correction performed.")
    end
end

function changeInterleavesToDynamics!(r::ReconParams{KdataPreprocessed{T}, <:Cartesian3D}) where T<:AbstractFloat
    numRepPerDyn = 1

    kdata = zeros(Complex{T}, size(r.data.kdata,1), size(r.data.kdata,2), numRepPerDyn*size(r.data.kdata,3), 
        size(r.data.kdata,4), Int((size(r.data.kdata,5))/numRepPerDyn*size(r.data.kdata,7)), size(r.data.kdata,6), 1)
    profileOrder = zeros(Int, 2, size(r.data.kdata,2), numRepPerDyn*size(r.data.kdata,3), size(r.data.kdata,4), 
        Int((size(r.data.kdata,5))/numRepPerDyn*size(r.data.kdata,7)), 1)
    if haskey(r.reconParameters, :motionCurveSorted)
        motionCurveSorted = zeros(T, size(kdata))
    end
    # for j in 1:Int(numDyn(r)-1/numRepPerDyn)
    for j in 1:Int(numDyn(r)/numRepPerDyn)
        for k in 1:size(r.data.kdata,7)
            # @assert numRepPerDyn == 2

            kdata[:,:,:,:,size(r.data.kdata,7)*(j-1)+k,:,:] .= r.data.kdata[:,:,:,:,j,:,k]
            profileOrder[:, :, :, :, size(r.data.kdata,7)*(j-1)+k, 1] .=  r.traj.profileOrder[:, :, :, :, j, k]
            if haskey(r.reconParameters, :motionCurveSorted)
                motionCurveSorted[:,:,:,:,size(r.data.kdata,7)*(j-1)+k,:,:] .= r.reconParameters[:motionCurveSorted][:,:,:,:,j,:,k]
            end
        end 
    end
    r.data = KdataPreprocessed(kdata)
    r.traj = Cartesian3D(profileOrder, :Cartesian3D)
    if haskey(r.reconParameters, :motionCurveSorted)
        r.reconParameters[:motionCurveSorted] = motionCurveSorted
    end
end

function softGatingWeights!(r::ReconParams{KdataPreprocessed{T}, <:Cartesian3D}) where T<:AbstractFloat    
    if haskey(r.reconParameters, :motionCurve)
        motionCurveSorted = permutedims(r.reconParameters[:motionCurveSorted], [1, 2, 3, 4, 5, 7, 6])
        weights = deepcopy(motionCurveSorted)
        motionCurve = r.reconParameters[:motionCurve]
        reference_state = motionCurve[r.reconParameters[:motionStatesCenter][1]]
        @show reference_state
        FWHM = motionCurve[r.reconParameters[:motionStatesCenter][3]] - motionCurve[r.reconParameters[:motionStatesCenter][1]]
        @show FWHM
        sigma = FWHM / 2.355
        scale = 2.0 * sigma.^2
        @show scale
        r.reconParameters[:softGatingWeights] = exp.(- 0.5 * (weights.-reference_state).^2 / sigma.^2)
        # state1 = (motionCurve[r.reconParameters[:motionStatesCenter][2]] + motionCurve[r.reconParameters[:motionStatesCenter][1]]) / 2
        # @show state1
    end
end

function subspaceBasis!(r::ReconParams{KdataPreprocessed{T}, <:Cartesian3D}) where T<:AbstractFloat    
    prep_dict = h5open(r.reconParameters[:prepDictPath])
    num_components = r.reconParameters[:iterativeReconParams][:subspaceComponents]
    mask_b0 = abs.(HDF5.attrs(prep_dict)["B0s"]) .<= 200 
    prep_dict = Array(prep_dict["dictionary"])
    prep_dict = prep_dict[:, :, :, :, :, mask_b0] ## Mask B0 values, maybe not needed
    prep_dict = reshape(prep_dict, prod(size(prep_dict)[1:3]), :)
    prep_dict = prep_dict'
    F = svd(prep_dict)
    basis = F.V[:,1:num_components]
    @show sum(F.S)
    r.reconParameters[:subspaceBasis] = basis
    r.reconParameters[:subspaceWeights] = F.S[1:num_components]#weights
end

function computeDensityCompensation!(r::ReconParams{KdataPreprocessed{T}, <:AbstractTrajectory}) where T<:AbstractFloat    
    if :sensMaps in keys(r.reconParameters)
        reconSize = size(r.reconParameters[:sensMaps])[1:3] 
    else
        reconSize = (size(r.data.kdata,1), r.reconParameters[:numKy], r.reconParameters[:numKz])
    end
    data = r.data.kdata
    dataTemp = permutedims(data, [1, 2, 3, 4, 5, 7, 6])
    profiles = deepcopy(r.traj.profileOrder)
    if :softGatingWeights in keys(r.reconParameters)
        weights = r.reconParameters[:softGatingWeights]
    else
        weights = nothing
    end

    subset_samples = zeros(Float32, size(dataTemp, 2), size(dataTemp, 3), size(dataTemp, 5))
    @views for k in 1:size(dataTemp,5)
        if weights !== nothing
            weights_temp = reshape(weights[1,:,:,1,k,1,1], :)
            mask_x = reshape(profiles[1,:,:,1,k,1], :)
            mask_y = reshape(profiles[2,:,:,1,k,1], :)
        end
        for j in 1:size(dataTemp,3)
            for i in 1:size(dataTemp,2)
                profile = profiles[:,i,j,1,k,1]
                if profile[1] > 0 && profile[2] > 0
                    if weights === nothing
                        subset_samples[i, j, k] = sum((profiles[1,:,:,1,k,1] .== profile[1]) .&& (profiles[2,:,:,1,k,1] .== profile[2])) 
                    else
                        mask = (mask_x .== profile[1]) .&& (mask_y .== profile[2])
                        subset_samples[i, j, k] = sum(weights_temp[mask])
                    end
                end
            end
        end
    end
    num_samples = repeat(reshape(subset_samples,:), size(dataTemp,1)*size(dataTemp,4)*size(dataTemp,6)*size(dataTemp,7))
    num_samples = reshape(num_samples, size(dataTemp,2), size(dataTemp,3), size(dataTemp,5), size(dataTemp,1), size(dataTemp,4), size(dataTemp,6), size(dataTemp,7))
    num_samples = permutedims(num_samples, [4, 1, 2, 5, 3, 6, 7])

    r.reconParameters[:sdcCartesian] = num_samples
    append!(r.performedMethods, [nameof(var"#self#")])
end

function sortData(r::ReconParams{KdataRaw{T}, <:Cartesian3D}; kykz_max=nothing) where T<:AbstractFloat
    card_phases = unique(r.data.labels[:card])
    if length(card_phases) > 1 && numInterleaves(r) == 1 ## Look-Locker
        @info("Look-Locker sequence assumed.")
        r.data.labels[:extr1] .= r.data.labels[:card]
        r.reconParameters[:LookLocker] = true
    elseif length(card_phases) > 1 && numInterleaves(r) > 1
        @error("Cardiac phases with multiple dynamics not supported!")
    else
        r.reconParameters[:LookLocker] = false
    end

    @debug("Sort Cartesian Spiral data.")
    labels = r.data.labels
    indices = accImagDataLabels(r)
    uniqueChan = unique(labels[:chan][indices])
    chan_indices = Dict([uniqueChan[i] => i for i in eachindex(axes(uniqueChan,1))])
    tfe_factor = round(Int, r.scanParameters[:TFEfactor])

    numMotionStates, _ = getBcurve(r)
    if haskey(r.reconParameters, :motionCurve) 
        # bCurve = repeat(r.reconParameters[:motionCurve]', numChan(r)*numEchoes(r)*tfe_factor) ## for motionStatesSelfGating function
        bCurve = repeat(r.reconParameters[:motionCurve]', numChan(r)*numEchoes(r))
        bCurve = reshape(bCurve,:)
    end
    num_dyn = numDyn(r)
    if r.reconParameters[:LookLocker]
        mask = (labels[:kz][indices].==0) .& (labels[:ky][indices].==0) .& (labels[:echo][indices].==0) .& (labels[:dyn][indices].==0) .& (labels[:chan][indices] .== unique(labels[:chan][indices])[1]) .& (labels[:extr1][indices].==1)
    else
        mask = (labels[:kz][indices].==0) .& (labels[:ky][indices].==0) .& (labels[:echo][indices].==0) .& (labels[:dyn][indices].==0) .& (labels[:chan][indices] .== unique(labels[:chan][indices])[1]) #.& (labels[:extr1][indices].==1)
    end
    # @show size(mask)
    # @show sum((labels[:kz][indices].==0) .& (labels[:ky][indices].==0))
    shots_labels = 1:sum(mask)
    # shots_labels = repeat(shots_labels', numChan(r)*numEchoes(r)*tfe_factor)
    # shots_labels = repeat(reshape(shots_labels, :), num_dyn*numInterleaves(r))
    shots_labels = repeat(shots_labels', numChan(r)*numEchoes(r)*tfe_factor*numInterleaves(r))
    shots_labels = repeat(reshape(shots_labels, :), num_dyn)

    mask = (labels[:kz][indices].==0) .& (labels[:ky][indices].==0) .& (labels[:echo][indices].==0) .& (labels[:chan][indices] .== unique(labels[:chan][indices])[1])
    virtual_echoes_label = repeat(1:tfe_factor, sum(mask))
    virtual_echoes_label = repeat(virtual_echoes_label', numChan(r)*numEchoes(r))

    data = zeros(Complex{T}, numKx(r), tfe_factor, maximum(shots_labels), numEchoes(r), 
        num_dyn, numChan(r), numInterleaves(r))
    min_ky = minimum(r.scanParameters[:KyRange]) 
    min_kz = minimum(r.scanParameters[:KzRange])
    r.reconParameters[:numKy] = Int(maximum(r.scanParameters[:KyRange]) - min_ky + 1)
    r.reconParameters[:numKz] = Int(maximum(r.scanParameters[:KzRange]) - min_kz + 1)
    if kykz_max !== nothing
        min_ky = -kykz_max[1]
        min_kz = -kykz_max[2]

        oldShape = collect([1, r.reconParameters[:numKy], r.reconParameters[:numKz]])'
        r.reconParameters[:numKy] = 2*kykz_max[1]+1
        r.reconParameters[:numKz] = 2*kykz_max[2]+1
        newShape = collect([1, r.reconParameters[:numKy], r.reconParameters[:numKz]])'
        @show newShape
        fac = newShape ./ oldShape 
        r.scanParameters[:encodingSize] = round.(Int, vec(fac) .* vec(r.scanParameters[:encodingSize]))
        r.scanParameters[:RecVoxelSize] = r.scanParameters[:AcqVoxelSize] ./ fac
        newShapeCsm = Tuple(round.(Int, size(r.reconParameters[:sensMaps]).*[fac..., 1.0]))
        r.reconParameters[:sensMaps] = imresize_dim(r.reconParameters[:sensMaps], newShapeCsm, dims=[1,2,3])
    end

    profiles = zeros(Int, 2,tfe_factor, maximum(shots_labels), numEchoes(r), 
        num_dyn,  numInterleaves(r))
    if haskey(r.reconParameters, :motionCurve) 
        motionCurveSorted = zeros(T, size(data))
    end
    ky_labels = labels[:ky][indices]
    kz_labels = labels[:kz][indices]
    chan_labels = labels[:chan][indices]
    echo_labels = labels[:echo][indices]
    dyn_labels = labels[:dyn][indices]
    inter_labels = labels[:extr1][indices]
    for i in eachindex(axes(r.data.accImagData, 2))
        current_chan = chan_labels[i]
        ky = ky_labels[i] - min_ky + 1
        kz = kz_labels[i] - min_kz + 1
        if kykz_max !== nothing
            if ky < 1 || kz < 1
                continue
            end
            if ky > 2*kykz_max[1]+1 || kz > 2*kykz_max[2]+1
                continue
            end
        end
        chan = chan_indices[current_chan]
        echo = echo_labels[i] + 1
        dyn = dyn_labels[i] + 1
        inter = inter_labels[i] + 1
        shot = shots_labels[i]
        if numMotionStates > 1
            motion_state = bCurve[i]
            @warn("Motion state option not supported at the moment.")
        else
            motion_state = 1
        end
        data[:, virtual_echoes_label[i], shot, echo, dyn, chan, inter] .= r.data.accImagData[:,i]
        profiles[1, virtual_echoes_label[i], shot, echo, dyn, inter] = ky#Tuple([ky, kz])
        profiles[2, virtual_echoes_label[i], shot, echo, dyn, inter] = kz#Tuple([ky, kz])
        if haskey(r.reconParameters, :motionCurve) 
            motionCurveSorted[:, virtual_echoes_label[i], shot, echo, dyn, chan, inter, motion_state] .= bCurve[i] #exp.(-x.^2/0.2)#1.0 - abs(bCurve2[i])
        end
    end
    if haskey(r.reconParameters, :motionCurve) 
        r.reconParameters[:motionCurveSorted] = motionCurveSorted 
    end
    append!(r.performedMethods, [nameof(var"#self#")])
    return ReconParams(r.filename, r.pathProc, r.scanParameters, 
        r.reconParameters, KdataPreprocessed(data), Cartesian3D(profiles, :CartesianSpiral), 
        r.performedMethods, r.imgData)
end
