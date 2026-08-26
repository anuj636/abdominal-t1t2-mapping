using Pkg
Pkg.activate(".") 
using ReconBMRR
using JLD2, CodecZlib

filename = "data/phantom/phantom_3_5mm.jld2"

@info "Loading raw data..."
r = jldopen(filename)["r"]
r.pathProc = dirname(replace(filename, "exp_raw" => "exp_pro"))
if !isdir(r.pathProc)
    mkpath(r.pathProc)
end

@info "Running preprocessing..."
phaseCorrDataBipolar!(r)
r2 = sortData(r)
applyPhaseCorrDataBipolar!(r2)
noisePreWhitening!(r2)
changeInterleavesToDynamics!(r2)

@info "Preprocessing complete!"
@info "Output path: $(r2.pathProc)"

if isdir(r2.pathProc)
    @info "✓ Output directory created successfully"
else
    @error "✗ Output directory not created"
end

@info "Test completed successfully!"
