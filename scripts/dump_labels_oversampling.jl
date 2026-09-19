# Dumps label vectors (ky/kz/chan/echo/dyn/extr1) and oversampling/geometry
# metadata from a raw KdataRaw JLD2, for cross-dataset comparison.
# Usage: julia --project=. scripts/dump_labels_oversampling.jl <jld2_path> <out_prefix>
using Pkg
Pkg.activate(".")
using JLD2, CodecZlib
using ReconBMRR

filename = length(ARGS) >= 1 ? ARGS[1] : error("usage: julia dump_labels_oversampling.jl <jld2> <out_prefix>")
out_prefix = length(ARGS) >= 2 ? ARGS[2] : "label_dump"

r = jldopen(filename)["r"]

println("=== dataset: $filename ===")
println("data type: ", typeof(r.data).name.name)
println("raw kdata/accImagData size: ", hasproperty(r.data, :accImagData) ? size(r.data.accImagData) : size(r.data.kdata))

if hasproperty(r.data, :labels)
    labels = r.data.labels
    indices = ReconBMRR.accImagDataLabels(r)

    ky = labels[:ky][indices]
    kz = labels[:kz][indices]
    chan = labels[:chan][indices]
    echo = labels[:echo][indices]
    dyn = labels[:dyn][indices]
    inter = labels[:extr1][indices]

    function summarize(name, v)
        println("  $name: n=$(length(v)) min=$(minimum(v)) max=$(maximum(v)) unique=$(length(unique(v)))")
    end

    println("=== label summary ===")
    summarize("ky", ky)
    summarize("kz", kz)
    summarize("chan", chan)
    summarize("echo", echo)
    summarize("dyn", dyn)
    summarize("extr1(inter)", inter)

    n_dump = min(200, length(ky))
    open("$(out_prefix)_labels.csv", "w") do io
        println(io, "idx,ky,kz,chan,echo,dyn,inter")
        for i in 1:n_dump
            println(io, "$i,$(ky[i]),$(kz[i]),$(chan[i]),$(echo[i]),$(dyn[i]),$(inter[i])")
        end
    end
    println("Wrote $(out_prefix)_labels.csv with $n_dump rows (of $(length(ky)) total)")
else
    println("No .labels field present (data already preprocessed) -- skipping label dump.")
end

println("=== oversampling / geometry scanParameters ===")
for k in [:KxOversampling, :KyOversampling, :KzOversampling, :SENSEExtraOversampling,
          :encodingSize, :AcqVoxelSize, :RecVoxelSize, :FOV, :curFOV,
          :KxRange, :KyRange, :KzRange, :TFEfactor, :XRes, :YRes, :ZRes, :TE_s]
    println("  $k = ", get(r.scanParameters, k, missing))
end
println("All scanParameters keys: ", collect(keys(r.scanParameters)))
