export CuCasprSubspaceOp

function cuprod_caspr_subspace!(y, x, profiles, basis, reconSize, numContr, numChan, numTFE, numShots, numBasis, numEchoes, weights)
    threads_per_block = 6
    threads_per_blockTFE = minimum([threads_per_block, numTFE])
    threads_per_blockShots = minimum([threads_per_block, numShots])
    threads_per_blockKx = minimum([threads_per_block, reconSize[1]])
    blocks_per_grid_x = (numTFE + threads_per_blockTFE - 1) ÷ threads_per_blockTFE
    blocks_per_grid_y = (numShots + threads_per_blockShots - 1) ÷ threads_per_blockShots
    blocks_per_grid_z = (reconSize[1] + threads_per_blockKx - 1) ÷ threads_per_blockKx
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    threads_per_block = (threads_per_blockTFE, threads_per_blockShots, threads_per_blockKx)  

    x = reshape(x,reconSize[1], reconSize[2], reconSize[3], numEchoes, numBasis, numChan) 
    fill!(y, 0)
    y = reshape(y,reconSize[1], numTFE, numShots, numEchoes, numContr, numChan)

    function prod_kernel!(profiles, x, y, numTFE, numShots, numKx, basis, weights)
        tfe_id = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
        shot_id = (blockIdx().y - 1) * blockDim().y  + (threadIdx().y)
        kx_id = (blockIdx().z - 1) * blockDim().z  + (threadIdx().z)
        if tfe_id <= numTFE && shot_id <= numShots && kx_id <= numKx
            for k in 1:size(y,5)
                for i in 1:size(y,4)
                    profile_x = profiles[1,tfe_id,shot_id,i,k]
                    profile_y = profiles[2,tfe_id,shot_id,i,k]
                    if profile_x > 0 && profile_y > 0
                        for j in 1:size(y,6)
                            for l in 1:size(basis,2)#-1
                                y[kx_id,tfe_id,shot_id,i,k,j] += x[kx_id,profile_x,profile_y,i,l,j] * conj(basis[tfe_id+numTFE*(k-1),l]) #* weights[l]
                                # y[kx_id,tfe_id,shot_id,i,k,j] += x[kx_id,profile_x,profile_y,i,l,j] * conj(basis[k,l]) #* weights[l]
                            end
                        end
                    end
                end
            end
        end
    end

    @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(profiles, x, y, numTFE, numShots, reconSize[1], basis, weights)
end

function cuctprod_caspr_subspace!(y, x, profiles, basis, reconSize, numContr, numChan, numTFE, numShots, numBasis, numEchoes, weights)
    threads_per_block = 8
    threads_per_blockChan = minimum([threads_per_block, numChan])
    threads_per_blockKx = minimum([threads_per_block, reconSize[1]])
    threads_per_blockContr = minimum([threads_per_block, numEchoes])
    blocks_per_grid_x = (numChan + threads_per_blockChan - 1) ÷ threads_per_blockChan
    blocks_per_grid_y = (reconSize[1] + threads_per_blockKx - 1) ÷ threads_per_blockKx
    blocks_per_grid_z = (numEchoes + threads_per_blockContr - 1) ÷ threads_per_blockContr
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)
    threads_per_block = (threads_per_blockChan, threads_per_blockKx, threads_per_blockContr)  

    x = reshape(x,reconSize[1], numTFE, numShots, numEchoes, numContr, numChan)
    fill!(y, 0)
    y = reshape(y,reconSize[1], reconSize[2], reconSize[3], numEchoes, numBasis, numChan)

    function prod_kernel!(profiles, x, y, numChan, numKx, numEchoes, basis, weights)
        chan_id = (blockIdx().x - 1) * blockDim().x  + (threadIdx().x)
        kx_id = (blockIdx().y - 1) * blockDim().y  + (threadIdx().y)
        echo_id = (blockIdx().z - 1) * blockDim().z  + (threadIdx().z)
        if chan_id <= numChan && kx_id <= numKx && echo_id <= numEchoes
            for i in 1:size(x,2)
                for j in 1:size(x,3)
                    for k in 1:size(x,5)
                        profile_x = profiles[1,i,j,echo_id,k]
                        profile_y = profiles[2,i,j,echo_id,k]
                        if profile_x > 0 && profile_y > 0
                            sample = x[kx_id,i,j,echo_id,k,chan_id]
                            for l in 1:size(basis,2)
                                y[kx_id,profile_x,profile_y,echo_id,l,chan_id] += sample * basis[i+size(x,2)*(k-1),l] #* 1/weights[l]
                                # y[kx_id,profile_x,profile_y,echo_id,l,chan_id] += sample * basis[k,l] #* 1/weights[l]
                            end
                        end
                    end
                end
            end
        end
    end

    @cuda blocks=blocks_per_grid threads=threads_per_block prod_kernel!(profiles, x, y, numChan, reconSize[1], numEchoes, basis, weights)
end

function CuCasprSubspaceOp(profiles, basis, reconSize, numContr, numChan, weights) 
    _, numTFE, numShots, numEchoesContr = size(profiles)
  numEchoes = Int(numEchoesContr/numContr)
  numBasis = size(basis,2)

  profiles_gpu = reshape(CuArray(profiles), 2, numTFE, numShots, numEchoes, numContr)
  basis_gpu = CuArray(basis)
  weights_gpu = CuArray(basis)

  return LinearOperator{ComplexF32}(reconSize[1]*prod(size(profiles)[2:3])*numEchoes*numContr*numChan, prod(reconSize)*numEchoes*numBasis*numChan, false, false,
                       (res,x) -> cuprod_caspr_subspace!(res, x, profiles_gpu, basis_gpu, reconSize, numContr, numChan, numTFE, numShots, numBasis, numEchoes, weights_gpu),
                       nothing,
                       (res,x) -> cuctprod_caspr_subspace!(res, x, profiles_gpu, basis_gpu, reconSize, numContr, numChan, numTFE, numShots, numBasis, numEchoes, weights_gpu),
                       S=S=CuArray{ComplexF32, 1, CUDA.Mem.DeviceBuffer})
end