
export CuCompositeOp, CompositeOp

mutable struct CompositeOp{T,U,V} <: AbstractLinearOperator{T}
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
  isWeighting :: Bool
  A::U
  B::V
  tmp::Vector{T}
end

"""
    CompositeOp(ops :: AbstractLinearOperator...)

composition/product of two Operators. Differs with * since it can handle normal operator
"""
function CompositeOp(A,B;isWeighting=false)
  nrow = A.nrow
  ncol = B.ncol
  S = promote_type(eltype(A), eltype(B))
  tmp_ = Vector{S}(undef, B.nrow)

  function produ!(res, x::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, B, x)
    return mul!(res, A, tmp)
  end

  function tprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, transpose(A), y)
    return mul!(res, transpose(B), tmp)
  end

  function ctprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, adjoint(A), y)
    return mul!(res, adjoint(B), tmp)
  end

  Op = CompositeOp( nrow, ncol, false, false,
                     (res,x) -> produ!(res,x,tmp_),
                     (res,y) -> tprodu!(res,y,tmp_),
                     (res,y) -> ctprodu!(res,y,tmp_), 
                     0, 0, 0, false, false, false, S[], S[],
                     isWeighting, A, B, tmp_)

  return Op
end

LinearOperators.storage_type(op::CompositeOp) = typeof(op.Mv5)

"""
∘(A::T1, B::T2; isWeighting::Bool=false) where {T1<:AbstractLinearOperator, T2<:AbstractLinearOperator}

  composition of two operators.
"""
function Base.:∘(A::T1, B::T2; isWeighting::Bool=false) where {T1<:AbstractLinearOperator, T2<:AbstractLinearOperator}
  return CompositeOp(A,B;isWeighting=isWeighting)
end

function Base.copy(S::CompositeOp{T}) where T
  A = copy(S.A)
  B = copy(S.B)
  return CompositeOp(A,B; isWeighting=S.isWeighting)
end

mutable struct CuCompositeOp{T,U,V,S} <: AbstractLinearOperator{T}
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
    isWeighting :: Bool
    A::U
    B::V
    tmp::S
end

function CuCompositeOp(A,B;isWeighting=false)
    nrow = A.nrow
    ncol = B.ncol
    S = promote_type(LinearOperators.storage_type(A), LinearOperators.storage_type(B))
    Mv5, Mtu5 = S(undef, 0), S(undef, 0)
    # tmp_ = Vector{S}(undef, B.nrow)
    tmp_ = S(undef, B.nrow)
  
    function produ!(res, x::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
      mul!(tmp, B, x)
      return mul!(res, A, tmp)
    end
  
    function tprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
      mul!(tmp, transpose(A), y)
      return mul!(res, transpose(B), tmp)
    end
  
    function ctprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
      mul!(tmp, adjoint(A), y)
      return mul!(res, adjoint(B), tmp)
    end
    type = promote_type(eltype(A), eltype(B))
    Op = CuCompositeOp{type, typeof(A), typeof(B), S}( nrow, ncol, false, false,
                       (res,x) -> produ!(res,x,tmp_),
                       (res,y) -> tprodu!(res,y,tmp_),
                       (res,y) -> ctprodu!(res,y,tmp_), 
                       0, 0, 0, false, false, false, Mv5, Mtu5,
                       isWeighting, A, B, tmp_)
  
    return Op
end
  
LinearOperators.storage_type(op::CuCompositeOp) = typeof(op.Mv5)

# function Base.:∘(A::T1, B::T2; isWeighting::Bool=false) where {T1<:AbstractLinearOperator, T2<:AbstractLinearOperator}
#     return CuCompositeOp(A,B;isWeighting=isWeighting)
# end
  
function Base.copy(S::CuCompositeOp{T}) where T
    A = copy(S.A)
    B = copy(S.B)
    return CuCompositeOp(A,B; isWeighting=S.isWeighting)
end