using JLD2

if length(ARGS) < 1
    error("Usage: julia --project=. scripts/preflight_recon.jl <jld2_path>")
end

path = ARGS[1]
r = jldopen(path, "r") do f
    f["r"]
end

function get_kdata_summary(data)
    if hasproperty(data, :kdata)
        kd = size(data.kdata)
        return kd, string(typeof(data)), :preprocessed
    end

    if hasproperty(data, :accImagData) && hasproperty(data, :labels)
        labels = data.labels
        acc_idx = haskey(labels, :LabelLookupTable) ? Int.(vec(labels[:LabelLookupTable][1])) : eachindex(labels[:ky])
        kd = (
            size(data.accImagData, 1),
            length(unique(labels[:ky][acc_idx])),
            length(unique(labels[:kz][acc_idx])),
            length(unique(labels[:echo][acc_idx])),
            length(unique(labels[:dyn][acc_idx])),
            length(unique(labels[:chan][acc_idx])),
            haskey(labels, :extr1) ? length(unique(labels[:extr1][acc_idx])) : 1,
        )
        return kd, string(typeof(data)), :raw
    end

    error("Preflight failed: unsupported data layout $(typeof(data)).")
end

te = Float32.(vec(get(r.scanParameters, :TE_s, Float32[])))
enc = Int.(vec(get(r.scanParameters, :encodingSize, Int32[0, 0, 0])))
rec = Float32.(vec(get(r.scanParameters, :RecVoxelSize, Float32[])))
fov = Float32.(vec(get(r.scanParameters, :FOV, Float32[])))
kd, data_type, data_layout = get_kdata_summary(r.data)

if length(te) < 2
    error("Preflight failed: TE_s must contain 2 echoes, got $(length(te)).")
end
if length(enc) < 3 || any(enc[1:3] .<= 1)
    error("Preflight failed: invalid encodingSize=$(enc).")
end
if length(rec) < 3 || any(rec[1:3] .<= 0)
    error("Preflight failed: invalid RecVoxelSize=$(rec).")
end
if length(fov) < 3 || any(fov[1:3] .<= 0)
    error("Preflight failed: invalid FOV=$(fov).")
end

derived = round.(Int, fov[1:3] ./ rec[1:3])
if any(abs.(derived .- enc[1:3]) .> 1)
    error("Preflight failed: FOV/RecVoxelSize mismatch. derived=$(derived) encodingSize=$(enc).")
end
if any(Tuple(kd[1:3]) .<= 1)
    error("Preflight failed: degenerate kdata spatial size=$(kd).")
end

println("[PREFLIGHT] scan metadata check passed: data_type=$(data_type), layout=$(data_layout), TE_s=$(te), encodingSize=$(enc), FOV=$(fov), RecVoxelSize=$(rec), kdata_size=$(kd)")