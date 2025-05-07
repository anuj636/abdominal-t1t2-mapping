export L1RegularizationDyn
import RegularizedLeastSquares: prox!

struct L1RegularizationDyn{T} <: AbstractParameterizedRegularization{T}
    λ::T
    weights
    reconSize
    numDyn::Int64
end
L1RegularizationDyn(λ, weights, reconSize; numDyn::Int64=1, kargs...) = L1RegularizationDyn(λ, weights, reconSize, numDyn)
  
function prox!(reg::L1RegularizationDyn, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
    if reg.numDyn == 1
        ε = eps(T)
        x .= max.((abs.(x).-λ),0) .* (x.+ε)./(abs.(x).+ε)
    else
      ε = eps(T)
      reconSize = reg.reconSize
      weights = repeat(reg.weights, inner = Int(reg.numDyn/size(reg.weights,1)))
      @assert length(reconSize) == 3
      for dim in 1:3
        mask_dim_1 = zeros(Bool, (reconSize[1] - 1) * reconSize[2] * reconSize[3] * reg.numDyn)
        mask_dim_2 = zeros(Bool, reconSize[1] * (reconSize[2] - 1) * reconSize[3] * reg.numDyn)
        mask_dim_3 = zeros(Bool, reconSize[1] * reconSize[2] * (reconSize[3] - 1) * reg.numDyn)
        if dim == 1
          mask_dim_1 .= true
        elseif dim == 2
          mask_dim_2 .= true
        else
          mask_dim_3 .= true
        end
        mask = vcat(mask_dim_1, mask_dim_2, mask_dim_3)
  
        x_ = reshape(x[mask], :, reg.numDyn)
        for i = 1:size(x_,2)
          weight = weights[i] / weights[1]
          x_[:,i] .= max.((abs.(x_[:,i]).-λ/weight),0) .* (x_[:,i].+ε)./(abs.(x_[:,i]).+ε) 
        end
        x[mask] = reshape(x_, :)
      end 
    end
    return x
end