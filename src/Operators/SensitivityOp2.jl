export SensitivityOp2, CuSensitivityOp2

function prod_smap!(y::AbstractVector{T}, smaps::AbstractMatrix{T}, x::AbstractVector{T}, numVox, numChan, numContr=1) where T
  x_ = reshape(x,numVox,numContr)
  y_ = reshape(y,numVox,numContr,numChan)

  @assert size(smaps) == (size(y_,1), size(y_,3))

  @inbounds for i ∈ CartesianIndices(y_)
    y_[i] = x_[i[1],i[2]] * smaps[i[1],i[3]]
  end
  return y
end

function ctprod_smap!(y::AbstractVector{T}, smapsC::AbstractMatrix{T}, x::AbstractVector{T}, numVox, numChan, numContr=1) where T
  x_ = reshape(x,numVox,numContr,numChan)
  y_ = reshape(y,numVox,numContr)

  @assert size(smapsC) == (size(x_,1), size(x_,3))

  y_ .= 0
  @inbounds for i ∈ CartesianIndices(x_)
    y_[i[1],i[2]] += x_[i] * smapsC[i[1],i[3]]
  end
  return y
end


"""
  SensitivityOp2(sensMaps::AbstractMatrix{T}, numContr=1)

builds a `LinearOperator` which performs multiplication of a given image with
the coil sensitivities specified in `sensMaps`
Use old version of MRIOperators.SensitivityOp due to different order of numContr and numChan

# Arguments
* `sensMaps`    - sensitivity maps ( 1. dim -> voxels, 2. dim-> coils)
* `numEchoes`   - number of contrasts to which the operator will be applied
"""
function SensitivityOp2(sensMaps::AbstractMatrix{T}, numContr=1) where T
    numVox, numChan = size(sensMaps)
    sensMapsC = conj.(sensMaps)
    return LinearOperator{T}(numVox*numContr*numChan, numVox*numContr, false, false,
                         (res,x) -> prod_smap!(res,sensMaps,x,numVox,numChan,numContr),
                         nothing,
                         (res,x) -> ctprod_smap!(res,sensMapsC,x,numVox,numChan,numContr))
end

"""
  SensitivityOp2(sensMaps::AbstractArray{T,4}, numContr=1)

builds a `LinearOperator` which performs multiplication of a given image with
the coil sensitivities specified in `sensMaps`
Use old version of MRIOperators.SensitivityOp due to different order of numContr and numChan

# Arguments
* `sensMaps`  - sensitivity maps ( 1.-3. dim -> voxels, 4. dim-> coils)
* `numContr`  - number of contrasts to which the operator will be applied
"""
function SensitivityOp2(sensMaps::AbstractArray{T,4}, numContr=1) where T #{T,D}
  sensMaps_mat = reshape(sensMaps, div(length(sensMaps),size(sensMaps,4)),size(sensMaps,4))
  return SensitivityOp2(sensMaps_mat, numContr)
end

function cu_prod_smap!(y::CuArray{T}, smaps::CuArray{T}, x::CuArray{T}, numVox, numChan, numContr=1) where T
  threads_per_block = 256
  blocks_per_grid = (numVox + threads_per_block - 1) ÷ threads_per_block

  x_ = reshape(x,numVox,numContr)
  y_ = reshape(y,numVox,numContr,numChan)


  function prod_kernel!(y, smaps, x, numVox, numChan, numContr)
      voxel_idx = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
      if voxel_idx <= numVox
          for contr_idx in 1:numContr
              for chan_idx in 1:numChan
                  y[voxel_idx, contr_idx, chan_idx] = x[voxel_idx, contr_idx] * smaps[voxel_idx, chan_idx]
              end
          end
      end
  
  end

  @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(y_, smaps, x_, numVox, numChan, numContr)
end

function cu_ctprod_smap!(y::CuArray{T}, smapsC::CuArray{T}, x::CuArray{T}, numVox, numChan, numContr=1) where T
  threads_per_block = 256
  blocks_per_grid = (numVox + threads_per_block - 1) ÷ threads_per_block

  x_ = reshape(x,numVox,numContr,numChan)
  y_ = reshape(y,numVox,numContr)


  function prod_kernel!(y, smapsC, x, numChan, numContr)
      voxel_idx = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
      if voxel_idx <= numVox
          for contr_idx in 1:numContr
              y[voxel_idx, contr_idx] = zero(T)
              for chan_idx in 1:numChan
                  y[voxel_idx, contr_idx] += x[voxel_idx, contr_idx, chan_idx] * smapsC[voxel_idx, chan_idx]
              end
          end
      end
  
  end

  @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(y_, smapsC, x_, numChan, numContr)
end


function CuSensitivityOp2(sensMaps::AbstractMatrix{T}, numContr=1) where T
  numVox, numChan = size(sensMaps)
  sensMapsC = conj.(sensMaps)

  cu_sensMaps = CuArray(sensMaps)
  cu_sensMapsC = CuArray(sensMapsC)

  tmp = CuArray(zeros(T, 1, 1))
  return LinearOperator{T}(numVox*numContr*numChan, numVox*numContr, false, false,
                       (res,x) -> cu_prod_smap!(res,cu_sensMaps,x,numVox,numChan,numContr),
                       nothing,
                       (res,x) -> cu_ctprod_smap!(res,cu_sensMapsC,x,numVox,numChan,numContr),
                       S=LinearOperators.storage_type(tmp))
end