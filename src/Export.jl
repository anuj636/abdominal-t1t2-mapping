export saveasImDataParams

function get_file(r::ReconParams, path, name, tag)
    filename = r.filename
    fsplit = split(basename(filename), '_')
    # Standard Philips raw filename has at least 5 underscore-separated parts
    # with fsplit[2] being an 8-digit date (DDMMYYYY) and fsplit[3] a time string.
    # Fall back to a safe fileID for non-standard filenames (e.g. from mrtools).
    if length(fsplit) >= 5 && length(fsplit[2]) >= 8 && length(fsplit[3]) >= 1
        datestr = string(fsplit[2][5:8], fsplit[2][3:4], fsplit[2][1:2])
        timestr = fsplit[3][1:end-1]
        SeriesNumber = string(fsplit[4], '0', fsplit[5])
        fileID = string(datestr, '_', timestr, '_', SeriesNumber)
    else
        # Use basename of filename as fileID for non-standard filenames
        fileID = replace(basename(filename), r"[^\w]" => "_")
    end
    if path == ""
        path = r.pathProc
    end
    # Create hdf5 fileID
    if name == ""
        newFilename = joinpath(path, string(fileID, "_", tag, ".h5"))
    else
        newFilename = joinpath(path, string(fileID, "_", tag, "_", name,".h5"))
    end
    fid = h5open(newFilename, "w")
    return fileID, fid, newFilename
end

function saveasImDataParams(r::ReconParams{<:AbstractKdata, <:AbstractTrajectory, ImgData{T}}; path::String="", name::String="", saveMotionState::Int=0) where T
    @debug("Save as ImDataParams.")
    fileID, fid, newFilename = get_file(r, path, name, "ImDataParamsBMRR")

    img = r.imgData.signal
    if haskey(r.reconParameters, :iSlice)
        iSlice = r.reconParameters[:iSlice]
        img = img[:,:,iSlice:iSlice,:,:]
    end
    if r.scanParameters[:AcqMode] == "Cartesian"
        # only if Cartesian sampling 
        img = reverse(img, dims=(2,3))
    elseif r.traj.name == :StackOfStars
        img = reverse(img, dims=(3))
    end

    g = create_group(fid, "ImDataParams")
    if haskey(r.scanParameters, :TE_s)
        nEchoes = length(r.scanParameters[:TE_s])
    elseif haskey(r.scanParameters, :TE)
        nEchoes = length(r.scanParameters[:TE])
    else
        nEchoes = 1
    end
    nEchoes = max(1, Int(nEchoes))
    nInter = 1 #r.scanParameters[:numInterleaves] TODO
    fov_vals = Float32.(vec(r.scanParameters[:FOV]))
    if length(fov_vals) >= 3
        fov3 = fov_vals[1:3]
    else
        # Keep metadata shape stable even when upstream FOV is malformed.
        fov3 = Float32.([size(img,1), size(img,2), size(img,3)])
    end
    HDF5.attributes(g)["voxelSize_mm"] = fov3 ./ Float32.([size(img,1), size(img,2), size(img,3)])

    if haskey(r.scanParameters, :FieldStrength)
        HDF5.attributes(g)["fieldStrength_T"] = r.scanParameters[:FieldStrength]
    else
        @warn "Missing scanParameters[:FieldStrength]; exporting fallback 0.0T"
        HDF5.attributes(g)["fieldStrength_T"] = 0.0f0
    end

    if haskey(r.scanParameters, :centerFreq_Hz)
        HDF5.attributes(g)["centerFreq_Hz"] = r.scanParameters[:centerFreq_Hz]
    else
        @warn "Missing scanParameters[:centerFreq_Hz]; exporting fallback 0.0 Hz"
        HDF5.attributes(g)["centerFreq_Hz"] = 0.0f0
    end

    if haskey(r.reconParameters, :subspaceBasis)
        g["subspaceBasis", deflate=3] = r.reconParameters[:subspaceBasis]
    end

    te_s_export = if haskey(r.scanParameters, :TE_s)
        Float32.(vec(r.scanParameters[:TE_s]))
    elseif haskey(r.scanParameters, :TE)
        te = Float32.(vec(r.scanParameters[:TE]))
        (length(te) > 0 && maximum(te) > 0.1f0) ? (te .* 1f-3) : te
    else
        Float32[]
    end
    if length(te_s_export) == 0
        @warn "Missing scanParameters[:TE_s] and [:TE]; exporting synthetic TE_s for $(nEchoes) echo(es)."
        te_s_export = Float32.(collect(1:nEchoes) .* 1f-3)
    end
    HDF5.attributes(g)["TE_s"] = te_s_export

    HDF5.attributes(g)["fileID"] = fileID  

    # Change dimension such that they match python toolbox
    s = size(img)
    if saveMotionState == 0
        img = reshape(img, s[1], s[2], s[3], nEchoes*nInter, :)
    else
        if r.reconParameters[:motionStatesRecon] == "all"
            numMotionStates = r.reconParameters[:motionGatingParams][:numClusters]
        else
            numMotionStates = parse(Int64, r.reconParameters[:motionStatesRecon])
        end
        img = reshape(img, s[1], s[2], s[3], nEchoes*nInter, :, numMotionStates)
        img = img[:,:,:,:,:,saveMotionState]
    end
    img = permutedims(img, [4,5,3,2,1])
    g["signal", deflate=3] = img
    
    process_relax_params!(r, fid)
    process_motion_params!(r, fid)

    close(fid)
    return newFilename
end

function process_motion_params!(r, fid)
    g = create_group(fid, "MotionParams")
    if haskey(r.reconParameters, :motionStatesCenter)
        g["motionCurve", deflate=3] = r.reconParameters[:motionCurve]
        g["motionStates", deflate=3] = r.reconParameters[:motionStates]
        g["motionStatesCenter"] = r.reconParameters[:motionStatesCenter]
        HDF5.attributes(g)["method"] = r.reconParameters[:motionMethod]
        HDF5.attributes(g)["status"] = "populated"
    else
        # Keep schema stable across environments, even when motion gating is disabled.
        g["motionCurve", deflate=3] = Float32[]
        g["motionStates", deflate=3] = Int32[]
        g["motionStatesCenter"] = Int32[]
        HDF5.attributes(g)["method"] = "none"
        HDF5.attributes(g)["status"] = "not_available"
    end
end

function process_relax_params!(r, fid)
    g = create_group(fid, "RelaxParams")
    if haskey(r.scanParameters, :isFINO) && r.scanParameters[:isFINO]
        if haskey(r.scanParameters, :FINO_TE_s)
            HDF5.attributes(g)["TE_s"] = r.scanParameters[:FINO_TE_s]
        end
        if haskey(r.scanParameters, :FINO_delay_s)
            HDF5.attributes(g)["delay_s"] = r.scanParameters[:FINO_delay_s]
        end
        if haskey(r.scanParameters, :FINO_angle_deg)
            HDF5.attributes(g)["angle_deg"] = r.scanParameters[:FINO_angle_deg]
        end
        if haskey(r.scanParameters, :FINO_startup_delay_s)
            HDF5.attributes(g)["startup_delay_s"] = r.scanParameters[:FINO_startup_delay_s]
        end
        HDF5.attributes(g)["status"] = "populated"
    else
        if haskey(r.scanParameters, :TE_s)
            HDF5.attributes(g)["TE_s"] = Float32.(vec(r.scanParameters[:TE_s]))
        elseif haskey(r.scanParameters, :TE)
            te = Float32.(vec(r.scanParameters[:TE]))
            HDF5.attributes(g)["TE_s"] = (length(te) > 0 && maximum(te) > 0.1f0) ? (te .* 1f-3) : te
        else
            HDF5.attributes(g)["TE_s"] = Float32[]
        end
        HDF5.attributes(g)["status"] = "not_available"
    end
end