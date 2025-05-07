export CuDiagOp

mutable struct DiagOp{T} <: AbstractLinearOperator{T}
    nrow :: Int
    ncol :: Int
    symmetric :: Bool
    hermitian :: Bool
    prod! :: Function
    tprod! :: Function
    ctprod! :: Function
    nprod :: Int
    ntprod :: Int
    nctprod :: Int
    args5 :: Bool
    use_prod5! :: Bool
    allocated5 :: Bool
    Mv5 :: Vector{T}
    Mtu5 :: Vector{T}
    ops
    equalOps :: Bool
    xIdx :: Vector{Int}
    yIdx :: Vector{Int}
  end
  
LinearOperators.storage_type(op::DiagOp) = typeof(op.Mv5)

function DiagOp(ops :: AbstractLinearOperator...)
    nrow = 0
    ncol = 0
    S = eltype(ops[1])
    for i = 1:length(ops)
        nrow += ops[i].nrow
        ncol += ops[i].ncol
        S = promote_type(S, eltype(ops[i]))
    end
  
    xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
    yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))
    batch_size = 200
  
    Op = DiagOp{S}( nrow, ncol, false, false,
        (res,x) -> (diagOpProd(res,x,nrow,xIdx,yIdx,batch_size,ops...)),
        (res,y) -> (diagOpTProd(res,y,ncol,yIdx,xIdx,batch_size,ops...)),
        (res,y) -> (diagOpCTProd(res,y,ncol,yIdx,xIdx,batch_size,ops...)),
        0, 0, 0, false, false, false, S[], S[],
        [ops...], false, xIdx, yIdx)
    return Op
end
  
function DiagOp(op::AbstractLinearOperator, N=1)
    nrow = N*op.nrow
    ncol = N*op.ncol
    S = eltype(op)
    ops = [copy(op) for n=1:N]
  
    xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
    yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))
  
    Op = DiagOp{S}( nrow, ncol, false, false,
        (res,x) -> (diagOpProd(res,x,nrow,xIdx,yIdx,ops...)),
        (res,y) -> (diagOpTProd(res,y,ncol,yIdx,xIdx,ops...)),
        (res,y) -> (diagOpCTProd(res,y,ncol,yIdx,xIdx,ops...)),
        0, 0, 0, false, false, false, S[], S[],
        ops, true, xIdx, yIdx )
    return Op
end
 
function diagOpProd(y::AbstractVector{T}, x::AbstractVector{T}, nrow::Int, xIdx, yIdx, batch_size, ops :: AbstractLinearOperator...) where T
    # @floop for i=1:length(ops)
    # x_gpu = CuArray(x)
    # y_gpu = CuArray(y)
    # for i=1:length(ops)
    #     mul!(view(y_gpu,yIdx[i]:yIdx[i+1]-1), ops[i], view(x_gpu,xIdx[i]:xIdx[i+1]-1))
    # end
    # return copyto!(y, Array(y_gpu))
    n_ops = length(ops)

    max_x_size = 0
    max_y_size = 0
    for i in 1:batch_size:length(ops)
        i_end = min(i + batch_size - 1, length(ops))
        max_x_size = max(max_x_size, xIdx[i_end + 1] - xIdx[i])
        max_y_size = max(max_y_size, yIdx[i_end + 1] - yIdx[i])
    end

    x_gpu_batch = CUDA.zeros(T, max_x_size)
    y_gpu_batch = CUDA.zeros(T, max_y_size)
    
    # Loop over ops with batch processing
    for i in 1:batch_size:n_ops
        # Define the end of the batch ensuring it doesn't exceed n_ops
        i_end = min(i+batch_size-1, n_ops)

        # # Determine the sizes needed for x and y in this batch
        # x_size = xIdx[i_end+1] - xIdx[i]
        # y_size = yIdx[i_end+1] - yIdx[i]

        # # Allocate GPU memory for the whole batch
        # x_gpu_batch = CUDA.zeros(ComplexF32, x_size)
        # y_gpu_batch = CUDA.zeros(ComplexF32, y_size)

        # Copy whole batch of x to GPU
        copyto!(x_gpu_batch, x[xIdx[i]:xIdx[i_end+1]-1])
        
        # For each operation in the batch
        for j = i:i_end
            # Offset indices for local batch
            local_xIdx_start = xIdx[j] - xIdx[i] + 1
            local_xIdx_end   = xIdx[j+1] - xIdx[i]
            local_yIdx_start = yIdx[j] - yIdx[i] + 1
            local_yIdx_end   = yIdx[j+1] - yIdx[i]

            # Perform multiplication on GPU
            mul!(view(y_gpu_batch, local_yIdx_start:local_yIdx_end),
                 ops[j],
                 view(x_gpu_batch, local_xIdx_start:local_xIdx_end))
        end

        # Copy whole batch of y back to CPU
        @view(y[yIdx[i]:yIdx[i_end+1]-1]) .= Array(@view(y_gpu_batch[1:yIdx[i_end+1]-yIdx[i]]))
    end
end

function diagOpTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, batch_size, ops :: AbstractLinearOperator...) where T
    # @floop for i=1:length(ops)
    # x_gpu = CuArray(x)
    # y_gpu = CuArray(y)
    # for i=1:length(ops)
    #     mul!(view(y_gpu,yIdx[i]:yIdx[i+1]-1), transpose(ops[i]), view(x_gpu,xIdx[i]:xIdx[i+1]-1))
    # end
    # #return y
    # return copyto!(y, Array(y_gpu))
    n_ops = length(ops)
    
    max_x_size = 0
    max_y_size = 0
    for i in 1:batch_size:length(ops)
        i_end = min(i + batch_size - 1, length(ops))
        max_x_size = max(max_x_size, xIdx[i_end + 1] - xIdx[i])
        max_y_size = max(max_y_size, yIdx[i_end + 1] - yIdx[i])
    end

    x_gpu_batch = CUDA.zeros(T, max_x_size)
    y_gpu_batch = CUDA.zeros(T, max_y_size)

    # Loop over ops with batch processing
    for i in 1:batch_size:n_ops
        # Define the end of the batch ensuring it doesn't exceed n_ops
        i_end = min(i+batch_size-1, n_ops)

        # # Determine the sizes needed for x and y in this batch
        # x_size = xIdx[i_end+1] - xIdx[i]
        # y_size = yIdx[i_end+1] - yIdx[i]

        # # Allocate GPU memory for the whole batch
        # x_gpu_batch = CUDA.zeros(ComplexF32, x_size)
        # y_gpu_batch = CUDA.zeros(ComplexF32, y_size)

        # Copy whole batch of x to GPU
        copyto!(x_gpu_batch, x[xIdx[i]:xIdx[i_end+1]-1])
        
        # For each operation in the batch
        for j = i:i_end
            # Offset indices for local batch
            local_xIdx_start = xIdx[j] - xIdx[i] + 1
            local_xIdx_end   = xIdx[j+1] - xIdx[i]
            local_yIdx_start = yIdx[j] - yIdx[i] + 1
            local_yIdx_end   = yIdx[j+1] - yIdx[i]

            # Perform multiplication on GPU
            mul!(view(y_gpu_batch, local_yIdx_start:local_yIdx_end),
                 transpose(ops[j]),
                 view(x_gpu_batch, local_xIdx_start:local_xIdx_end))
        end

        # Copy whole batch of y back to CPU
        y[yIdx[i]:yIdx[i_end+1]-1] .= Array(y_gpu_batch[1:yIdx[i_end+1]-yIdx[i]])
    end
end

function diagOpCTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, batch_size, ops :: AbstractLinearOperator...) where T
    # # @floop for i=1:length(ops)
    # x_gpu = CuArray(x)
    # y_gpu = CuArray(y)
    # for i=1:length(ops)
    #     mul!(view(y_gpu,yIdx[i]:yIdx[i+1]-1), adjoint(ops[i]), view(x_gpu,xIdx[i]:xIdx[i+1]-1))
    # end
    # #return y
    # return copyto!(y, Array(y_gpu))
    n_ops = length(ops)

    max_x_size = 0
    max_y_size = 0
    for i in 1:batch_size:length(ops)
        i_end = min(i + batch_size - 1, length(ops))
        max_x_size = max(max_x_size, xIdx[i_end + 1] - xIdx[i])
        max_y_size = max(max_y_size, yIdx[i_end + 1] - yIdx[i])
    end

    x_gpu_batch = CUDA.zeros(T, max_x_size)
    y_gpu_batch = CUDA.zeros(T, max_y_size)
    
    # Loop over ops with batch processing
    for i in 1:batch_size:n_ops
        # Define the end of the batch ensuring it doesn't exceed n_ops
        i_end = min(i+batch_size-1, n_ops)

        # # Determine the sizes needed for x and y in this batch
        # x_size = xIdx[i_end+1] - xIdx[i]
        # y_size = yIdx[i_end+1] - yIdx[i]

        # # Allocate GPU memory for the whole batch
        # x_gpu_batch = CUDA.zeros(ComplexF32, x_size)
        # y_gpu_batch = CUDA.zeros(ComplexF32, y_size)

        # Copy whole batch of x to GPU
        copyto!(x_gpu_batch, x[xIdx[i]:xIdx[i_end+1]-1])
        
        # For each operation in the batch
        for j = i:i_end
            # Offset indices for local batch
            local_xIdx_start = xIdx[j] - xIdx[i] + 1
            local_xIdx_end   = xIdx[j+1] - xIdx[i]
            local_yIdx_start = yIdx[j] - yIdx[i] + 1
            local_yIdx_end   = yIdx[j+1] - yIdx[i]

            # Perform multiplication on GPU
            mul!(view(y_gpu_batch, local_yIdx_start:local_yIdx_end),
                 adjoint(ops[j]),
                 view(x_gpu_batch, local_xIdx_start:local_xIdx_end))
        end

        # Copy whole batch of y back to CPU
        y[yIdx[i]:yIdx[i_end+1]-1] .= Array(y_gpu_batch[1:yIdx[i_end+1]-yIdx[i]])
    end
end


mutable struct CuDiagOp{T, S} <: AbstractLinearOperator{T}
    nrow :: Int
    ncol :: Int
    symmetric :: Bool
    hermitian :: Bool
    prod! :: Function
    tprod! :: Function
    ctprod! :: Function
    nprod :: Int
    ntprod :: Int
    nctprod :: Int
    args5 :: Bool
    use_prod5! :: Bool
    allocated5 :: Bool
    Mv5 :: S
    Mtu5 :: S
    ops
    equalOps :: Bool
    xIdx :: Vector{Int}
    yIdx :: Vector{Int}
end
  
CuDiagOp(ops::AbstractLinearOperator...) = CuDiagOp(ops)

LinearOperators.storage_type(op::CuDiagOp) = typeof(op.Mv5)

function CuDiagOp(ops)
    nrow = 0
    ncol = 0
    S = LinearOperators.storage_type(ops[1])
    type = eltype(ops[1])
    for i = 1:length(ops)
      nrow += ops[i].nrow
      ncol += ops[i].ncol
      S = promote_type(S, LinearOperators.storage_type(ops[i]))
      type = promote_type(type, eltype(ops[i]))
    end
  
    xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
    yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))

    Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  
    Op = CuDiagOp{type, S}( nrow, ncol, false, false,
                       (res,x) -> (cuDiagOpProd(res,x,nrow,xIdx,yIdx,ops...)),
                       (res,y) -> (cuDiagOpTProd(res,y,ncol,yIdx,xIdx,ops...)),
                       (res,y) -> (cuDiagOpCTProd(res,y,ncol,yIdx,xIdx,ops...)),
                       0, 0, 0, false, false, false, Mv5, Mtu5,
                       [ops...], false, xIdx, yIdx)
  
    return Op
end
  
# function CuDiagOp(op::AbstractLinearOperator, N=1)
#     nrow = N*op.nrow
#     ncol = N*op.ncol
#     S = LinearOperators.storage_type(op)
#     ops = [op]
  
#     xIdx = cumsum(vcat(1,[ops[i].ncol for i=1:length(ops)]))
#     yIdx = cumsum(vcat(1,[ops[i].nrow for i=1:length(ops)]))

#     Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  
#     Op = MRIOperators.DiagOp{S}( nrow, ncol, false, false,
#                       (res,x) -> (cuDiagOpProd(res,x,nrow,xIdx,yIdx,ops...)),
#                       (res,y) -> (cuDiagOpTProd(res,y,ncol,yIdx,xIdx,ops...)),
#                       (res,y) -> (cuDiagOpCTProd(res,y,ncol,yIdx,xIdx,ops...)),
#                        0, 0, 0, false, false, false, Mv5, Mtu5,
#                        ops, true, xIdx, yIdx )
  
#     return Op
# end

function cuDiagOpProd(y::CuArray{T}, x::CuArray{T}, nrow::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    for i=1:length(ops)
      mul!(view(y,yIdx[i]:yIdx[i+1]-1), ops[i], view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end
  
function cuDiagOpTProd(y::CuArray{T}, x::CuArray{T}, ncol::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    for i=1:length(ops)
      mul!(view(y,yIdx[i]:yIdx[i+1]-1), transpose(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end
  
function cuDiagOpCTProd(y::CuArray{T}, x::CuArray{T}, ncol::Int, xIdx, yIdx, ops :: AbstractLinearOperator...) where T
    for i=1:length(ops)
      mul!(view(y,yIdx[i]:yIdx[i+1]-1), adjoint(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
    end
    return y
end