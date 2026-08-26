module BridgeDefaults

using ReconBMRR

function build_label_lookup_table(n::Int)
    return Any[
        reshape(Float64.(collect(1:n)), 1, :),  # accImagData indices
        reshape(Float64[], 1, 0),               # rejImagData indices
        reshape(Float64[], 1, 0),               # phaseCorrData indices
        reshape(Float64[], 1, 0),               # freqCorrData indices
        reshape(Float64[], 1, 0),               # noiseData indices
    ]
end

function default_labels(n::Int)
    return Dict{Symbol,Any}(
        :ky => zeros(Int32, n),
        :kz => zeros(Int32, n),
        :echo => zeros(UInt16, n),
        :dyn => zeros(UInt16, n),
        :chan => zeros(UInt16, n),
        :extr1 => zeros(UInt16, n),
        :card => zeros(UInt16, n),
        :sign => zeros(Int8, n),
        :typ => zeros(UInt8, n),
        :mix => zeros(UInt16, n),
        :LabelLookupTable => build_label_lookup_table(n),
    )
end

function default_scan_parameters()
    return Dict{Symbol,Any}(
        :AcqMode => "Cartesian",
        :FieldStrength => Float32(3.0),
        :TR => Float32(0.0),
        :FlipAngle => Float32(0.0),
        :TE => Float32[],
        :AcqVoxelSize => Float32[1.0, 1.0, 1.0],
        :RecVoxelSize => Float32[1.0, 1.0, 1.0],
        :FOV => Float32[1.0, 1.0, 1.0],
        :KxRange => Int32[],
        :KyRange => Int32[],
        :KzRange => Int32[],
        :encodingSize => Int32[1, 1, 1],
        :TFEfactor => 1,
    )
end

function default_recon_parameters(scan::Dict{Symbol,Any}=default_scan_parameters())
    return Dict{Symbol,Any}(
        :cuda => false,
        :cudaSolver => false,
        :useDoublePrecision => false,
        :artificalUndersampling => 0,
        :export => Dict{Symbol,Any}(),
        :iterativeReconParams => ReconBMRR.setIterativeReconParams(),
        :motionGating => false,
        :motionGatingParams => Dict{Symbol,Any}(),
        :motionStatesRecon => nothing,
        :noisePreWhitening => false,
        :upsampleRecVoxelSize => scan[:AcqMode] == "Radial" ? true : nothing,
        :removeOversampling => true,
        :kspOrdered => false,
        :coilSensitivities => false,
    )
end

function deep_merge(base::AbstractDict, override::AbstractDict; include_new_keys::Bool=true)
    merged = Dict{Symbol,Any}()
    for (k, v) in base
        merged[k] = deepcopy(v)
    end
    for (k, v_override) in override
        if haskey(merged, k) && merged[k] isa AbstractDict && v_override isa AbstractDict
            merged[k] = deep_merge(merged[k], v_override; include_new_keys=true)
        elseif haskey(merged, k) || include_new_keys
            merged[k] = deepcopy(v_override)
        end
    end
    return merged
end

end
