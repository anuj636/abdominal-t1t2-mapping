export CasprOp, CuCasprOp, CuCasprOpDiff

# ─── CPU implementation ────────────────────────────────────────────────────────

function prod_caspr!(y, x, profiles, reconSize, numContr, numChan, numTFE, numShots)
    x = reshape(x, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], numTFE, numShots, numContr, numChan)
    for k in 1:numContr
        for j in 1:numShots
            for i in 1:numTFE
                px = profiles[1, i, j, k]
                py = profiles[2, i, j, k]
                if px > 0 && py > 0
                    for c in 1:numChan
                        for kx in 1:reconSize[1]
                            y[kx, i, j, k, c] = x[kx, px, py, k, c]
                        end
                    end
                end
            end
        end
    end
end

function tprod_caspr!(y, x, profiles, reconSize, numContr, numChan, numTFE, numShots)
    x = reshape(x, reconSize[1], numTFE, numShots, numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)
    for k in 1:numContr
        for j in 1:numShots
            for i in 1:numTFE
                px = profiles[1, i, j, k]
                py = profiles[2, i, j, k]
                if px > 0 && py > 0
                    for c in 1:numChan
                        for kx in 1:reconSize[1]
                            y[kx, px, py, k, c] += x[kx, i, j, k, c]
                        end
                    end
                end
            end
        end
    end
end

function CasprOp(profiles, reconSize, numContr, numChan)
    _, numTFE, numShots, _ = size(profiles)
    nIn  = prod(reconSize) * numContr * numChan
    nOut = reconSize[1] * numTFE * numShots * numContr * numChan
    return LinearOperator{ComplexF32}(nOut, nIn, false, false,
        (res, x) -> prod_caspr!(res, x, profiles, reconSize, numContr, numChan, numTFE, numShots),
        nothing,
        (res, x) -> tprod_caspr!(res, x, profiles, reconSize, numContr, numChan, numTFE, numShots))
end

# ─── GPU implementation ────────────────────────────────────────────────────────

function cuprod_caspr!(y, x, profiles, reconSize, numContr, numChan, numTFE, numShots)
    tpb = 6
    tpb_tfe   = min(tpb, numTFE)
    tpb_shots = min(tpb, numShots)
    tpb_kx    = min(tpb, reconSize[1])
    bpg = (
        (numTFE      + tpb_tfe   - 1) ÷ tpb_tfe,
        (numShots    + tpb_shots - 1) ÷ tpb_shots,
        (reconSize[1] + tpb_kx   - 1) ÷ tpb_kx,
    )
    tpb_vec = (tpb_tfe, tpb_shots, tpb_kx)

    x = reshape(x, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], numTFE, numShots, numContr, numChan)

    function prod_kernel!(profiles, x, y, numTFE, numShots, numKx)
        tfe_id  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        shot_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        kx_id   = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        if tfe_id <= numTFE && shot_id <= numShots && kx_id <= numKx
            for k in 1:size(y, 4)
                px = profiles[1, tfe_id, shot_id, k]
                py = profiles[2, tfe_id, shot_id, k]
                if px > 0 && py > 0
                    for c in 1:size(y, 5)
                        y[kx_id, tfe_id, shot_id, k, c] = x[kx_id, px, py, k, c]
                    end
                end
            end
        end
        return
    end

    @cuda blocks=bpg threads=tpb_vec prod_kernel!(profiles, x, y, numTFE, numShots, reconSize[1])
end

function cuctprod_caspr!(y, x, profiles, reconSize, numContr, numChan, numTFE, numShots)
    tpb = 8
    tpb_chan  = min(tpb, numChan)
    tpb_kx    = min(tpb, reconSize[1])
    tpb_contr = min(tpb, numContr)
    bpg = (
        (numChan      + tpb_chan  - 1) ÷ tpb_chan,
        (reconSize[1] + tpb_kx   - 1) ÷ tpb_kx,
        (numContr     + tpb_contr - 1) ÷ tpb_contr,
    )
    tpb_vec = (tpb_chan, tpb_kx, tpb_contr)

    x = reshape(x, reconSize[1], numTFE, numShots, numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)

    function tprod_kernel!(profiles, x, y, numChan, numKx, numContr)
        chan_id  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        kx_id   = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        contr_id = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        if chan_id <= numChan && kx_id <= numKx && contr_id <= numContr
            for i in 1:size(x, 2)
                for j in 1:size(x, 3)
                    px = profiles[1, i, j, contr_id]
                    py = profiles[2, i, j, contr_id]
                    if px > 0 && py > 0
                        y[kx_id, px, py, contr_id, chan_id] += x[kx_id, i, j, contr_id, chan_id]
                    end
                end
            end
        end
        return
    end

    @cuda blocks=bpg threads=tpb_vec tprod_kernel!(profiles, x, y, numChan, reconSize[1], numContr)
end

function CuCasprOp(profiles, reconSize, numContr, numChan)
    _, numTFE, numShots, _ = size(profiles)
    profiles_gpu = CuArray(profiles)
    nIn  = prod(reconSize) * numContr * numChan
    nOut = reconSize[1] * numTFE * numShots * numContr * numChan
    return LinearOperator{ComplexF32}(nOut, nIn, false, false,
        (res, x) -> cuprod_caspr!(res, x, profiles_gpu, reconSize, numContr, numChan, numTFE, numShots),
        nothing,
        (res, x) -> cuctprod_caspr!(res, x, profiles_gpu, reconSize, numContr, numChan, numTFE, numShots),
        S=CuArray{ComplexF32, 1, CUDA.Mem.DeviceBuffer})
end

# ─── GPU implementation with TFE relaxation weighting ─────────────────────────
# Models the signal evolution along the TFE train due to repeated RF pulses.
# Each TFE step is weighted by the accumulated cos(FA)^(i-1) roll-off (spoiled
# GRE steady-state, T1 recovery between shots neglected for short TR).

function _tfe_weights(numTFE::Int, flipAngle_deg, TR_ms; T1_ms::Float32=1000f0)
    FA  = Float32(flipAngle_deg) * Float32(pi) / 180f0
    E1  = exp(-Float32(TR_ms) / T1_ms)
    cosFA_E1 = cos(FA) * E1
    # M_z(i) ∝ (cos(FA)*E1)^(i-1); normalise so first step = 1
    weights = [cosFA_E1^(i - 1) for i in 1:numTFE]
    return Float32.(weights)
end

function cuprod_caspr_diff!(y, x, profiles, tfe_weights, reconSize, numContr, numChan, numTFE, numShots)
    tpb = 6
    tpb_tfe   = min(tpb, numTFE)
    tpb_shots = min(tpb, numShots)
    tpb_kx    = min(tpb, reconSize[1])
    bpg = (
        (numTFE      + tpb_tfe   - 1) ÷ tpb_tfe,
        (numShots    + tpb_shots - 1) ÷ tpb_shots,
        (reconSize[1] + tpb_kx   - 1) ÷ tpb_kx,
    )
    tpb_vec = (tpb_tfe, tpb_shots, tpb_kx)

    x = reshape(x, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], numTFE, numShots, numContr, numChan)

    function prod_kernel!(profiles, tfe_weights, x, y, numTFE, numShots, numKx)
        tfe_id  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        shot_id = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        kx_id   = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        if tfe_id <= numTFE && shot_id <= numShots && kx_id <= numKx
            w = tfe_weights[tfe_id]
            for k in 1:size(y, 4)
                px = profiles[1, tfe_id, shot_id, k]
                py = profiles[2, tfe_id, shot_id, k]
                if px > 0 && py > 0
                    for c in 1:size(y, 5)
                        y[kx_id, tfe_id, shot_id, k, c] = w * x[kx_id, px, py, k, c]
                    end
                end
            end
        end
        return
    end

    @cuda blocks=bpg threads=tpb_vec prod_kernel!(profiles, tfe_weights, x, y, numTFE, numShots, reconSize[1])
end

function cuctprod_caspr_diff!(y, x, profiles, tfe_weights, reconSize, numContr, numChan, numTFE, numShots)
    tpb = 8
    tpb_chan  = min(tpb, numChan)
    tpb_kx    = min(tpb, reconSize[1])
    tpb_contr = min(tpb, numContr)
    bpg = (
        (numChan      + tpb_chan  - 1) ÷ tpb_chan,
        (reconSize[1] + tpb_kx   - 1) ÷ tpb_kx,
        (numContr     + tpb_contr - 1) ÷ tpb_contr,
    )
    tpb_vec = (tpb_chan, tpb_kx, tpb_contr)

    x = reshape(x, reconSize[1], numTFE, numShots, numContr, numChan)
    fill!(y, 0)
    y = reshape(y, reconSize[1], reconSize[2], reconSize[3], numContr, numChan)

    function tprod_kernel!(profiles, tfe_weights, x, y, numChan, numKx, numContr)
        chan_id  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        kx_id   = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        contr_id = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        if chan_id <= numChan && kx_id <= numKx && contr_id <= numContr
            for i in 1:size(x, 2)
                w = tfe_weights[i]
                for j in 1:size(x, 3)
                    px = profiles[1, i, j, contr_id]
                    py = profiles[2, i, j, contr_id]
                    if px > 0 && py > 0
                        y[kx_id, px, py, contr_id, chan_id] += w * x[kx_id, i, j, contr_id, chan_id]
                    end
                end
            end
        end
        return
    end

    @cuda blocks=bpg threads=tpb_vec tprod_kernel!(profiles, tfe_weights, x, y, numChan, reconSize[1], numContr)
end

function CuCasprOpDiff(profiles, reconSize, numContr, numChan, flipAngle, TR)
    _, numTFE, numShots, _ = size(profiles)
    profiles_gpu     = CuArray(profiles)
    tfe_weights_gpu  = CuArray(_tfe_weights(numTFE, flipAngle, TR))
    nIn  = prod(reconSize) * numContr * numChan
    nOut = reconSize[1] * numTFE * numShots * numContr * numChan
    return LinearOperator{ComplexF32}(nOut, nIn, false, false,
        (res, x) -> cuprod_caspr_diff!(res, x, profiles_gpu, tfe_weights_gpu, reconSize, numContr, numChan, numTFE, numShots),
        nothing,
        (res, x) -> cuctprod_caspr_diff!(res, x, profiles_gpu, tfe_weights_gpu, reconSize, numContr, numChan, numTFE, numShots),
        S=CuArray{ComplexF32, 1, CUDA.Mem.DeviceBuffer})
end
