getobs(data; obsdim = default_obsdim(data)) =
    __getobs(data, convert(LearnBase.ObsDimension,obsdim))

# These "__" methods serve the sole purpose of avoiding ambiguities
# when a user defines "getobs" for his/her type.
# Basically if obsdim is undefined, we assume there is no
# getobs(::MyType, ::ObsDimension) to care about.
# All others are fed back into "getobs" dispatch
@inline __getobs(data, obsdim::ObsDim.Undefined) = _getobs(data, obsdim)
@inline __getobs(data, obsdim::ObsDimension) = getobs(data, obsdim)
@inline __getobs(data::Tuple, obsdim::Tuple) = getobs(data, obsdim)

getobs!(buffer, data) = getobs(data)
getobs!(buffer, data, idx, obsdim) = getobs(data, idx, obsdim)
getobs!(buffer, data, idx; obsdim = default_obsdim(data)) =
    getobs!(buffer, data, idx, convert(LearnBase.ObsDimension,obsdim))
# NOTE: default to not use buffer since copy! may not be defined
# getobs!(buffer, data) = copy!(buffer, getobs(data))
# getobs!(buffer, data, idx, obsdim) = copy!(buffer, getobs(data, idx, obsdim))

# fallback methods discards unused obsdim
nobs(data, ::ObsDim.Undefined)::Int = nobs(data)
getobs(data, idx, ::ObsDim.Undefined) = getobs(data, idx)

# to accumulate indices as views instead of copies
_view(indices::Range, i::Int) = indices[i]
_view(indices::Range, i::Range) = indices[i]
_view(indices, i::Int) = indices[i] # to throw error in case
_view(indices, i) = view(indices, i)

"""
    nobs(data, [obsdim]) -> Int

Return the total number of observations contained in `data`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense for
the type of `data`. See `?LearnBase.ObsDim` for more information.
"""
function nobs(data; obsdim = default_obsdim(data))::Int
    nobsdim = convert(LearnBase.ObsDimension,obsdim)
    # make sure we don't bounce between fallback methods
    typeof(nobsdim) <: ObsDim.Undefined && throw(MethodError(nobs, (data,)))
    nobs(data, nobsdim)
end

"""
    getobs(data, [idx], [obsdim])

Return the observation(s) in `data` that correspond to the given
index/indices in `idx`. Note that `idx` can be of type `Int` or
`AbstractVector`.

The returned observation(s) should be in the form intended to be
passed as-is to some learning algorithm. There is no strict
requirement that dictates what form or type that is. We do,
however, expect it to be consistent for `idx` being an integer,
as well as `idx` being an abstract vector, respectively.

If it makes sense for the type of `data`, `obsdim` can be used to
specify which dimension of `data` denotes the observations. It
can be specified in a typestable manner as a positional argument
(see `?LearnBase.ObsDim`), or more conveniently as a smart
keyword argument.
"""
getobs(data, arg, args...; kw...) = _getobs(data, arg, args...; kw...)
getobs(data, arg) = _getobs(data, arg)

# These "_" methods serve the sole purpose of avoiding ambiguities
# when a user defines "getobs" for his/her type.
# The problem would be getobs(::Any, ::Union{ObsDimension,Tuple}),
# which would likely collide with getobs(::MyType, idx)
@inline function _getobs(data, idx; obsdim = default_obsdim(data))
    nobsdim = convert(LearnBase.ObsDimension,obsdim)
    # make sure we don't bounce between fallback methods
    typeof(nobsdim) <: ObsDim.Undefined && throw(MethodError(getobs, (data,idx)))
    getobs(data, idx, nobsdim)
end

@inline _getobs(data, obsdim::Union{ObsDimension,Tuple}) =
    getobs(data, 1:nobs(data,obsdim), obsdim)

# --------------------------------------------------------------------
# Arrays

nobs{DIM}(A::AbstractArray, ::ObsDim.Constant{DIM})::Int = size(A, DIM)
nobs{T,N}(A::AbstractArray{T,N}, ::ObsDim.Last)::Int = size(A, N)

getobs(A::Array, ::ObsDimension=default_obsdim(A)) = A
getobs(A::AbstractSparseArray, ::ObsDimension=default_obsdim(A)) = A
getobs(A::SubArray, ::ObsDimension=default_obsdim(A)) = copy(A)
getobs(A::AbstractArray, ::ObsDimension=default_obsdim(A)) = collect(A)

"""
    getobs!(buffer, data, [idx], [obsdim]) -> buffer

Write the observation(s) from `data` that correspond to the given
index/indices in `idx` into `buffer`. Note that `idx` can be of
type `Int` or `AbstractVector`.

Unless explicitly implemented for `data` it defaults to returning
`getobs(data, idx, obsdim)` in which case `buffer` is ignored.

If it makes sense for the type of `data`, `obsdim` can be used to
specify which dimension of `data` denotes the observations. It
can be specified in a typestable manner as a positional argument
(see `?LearnBase.ObsDim`), or more conveniently as a smart
keyword argument.
"""
getobs!(buffer, A::AbstractSparseArray, idx, obsdim) = getobs(A, idx, obsdim)
getobs!(buffer, A::AbstractSparseArray) = getobs(A)

getobs!(buffer, A::AbstractArray, idx, obsdim) = copy!(buffer, datasubset(A, idx, obsdim))
getobs!(buffer, A::AbstractArray) = copy!(buffer, A)

# catch the undefined setting for consistency.
# should never happen by accident
getobs(A::AbstractArray, idx, obsdim::ObsDim.Undefined) =
    throw(MethodError(getobs, (A, idx, obsdim)))

@generated function getobs{T,N}(A::AbstractArray{T,N}, idx, obsdim::ObsDimension)
    @assert N > 0
    if N == 1 && idx <: Integer
        :(A[idx])
    elseif obsdim <: ObsDim.First
        :(getindex(A, idx, $(fill(:(:),N-1)...)))
    elseif obsdim <: ObsDim.Last || (obsdim <: ObsDim.Constant && obsdim.parameters[1] == N)
        :(getindex(A, $(fill(:(:),N-1)...), idx))
    else # obsdim <: ObsDim.Constant
        DIM = obsdim.parameters[1]
        DIM > N && throw(DimensionMismatch("the given obsdim=$DIM is greater than the number of available dimensions N=$N"))
        :(getindex(A, $(fill(:(:),DIM-1)...), idx, $(fill(:(:),N-DIM)...)))
    end
end

# --------------------------------------------------------------------
# Tuples

_check_nobs_error() = throw(DimensionMismatch("all data container must have the same number of observations"))

function _check_nobs(tup::Tuple)
    length(tup) == 0 && return
    n1 = nobs(tup[1])
    for i=2:length(tup)
        nobs(tup[i]) != n1 && _check_nobs_error()
    end
end

function _check_nobs(tup::Tuple, obsdim::ObsDimension)
    length(tup) == 0 && return
    n1 = nobs(tup[1], obsdim)
    for i=2:length(tup)
        nobs(tup[i], obsdim) != n1 && _check_nobs_error()
    end
end

function _check_nobs(tup::Tuple, obsdims::Tuple)
    length(tup) == 0 && return
    length(tup) == length(obsdims) || throw(DimensionMismatch("number of elements in obsdim doesn't match data"))
    all(map(x-> typeof(x) <: Union{ObsDimension,Tuple}, obsdims)) || throw(MethodError(_check_nobs, (tup, obsdims)))
    n1 = nobs(tup[1], obsdims[1])
    for i=2:length(tup)
        nobs(tup[i], obsdims[i]) != n1 && _check_nobs_error()
    end
end

function nobs(tup::Tuple, ::ObsDim.Undefined = ObsDim.Undefined())::Int
    _check_nobs(tup)
    length(tup) == 0 ? 0 : nobs(tup[1])
end

function nobs(tup::Tuple, obsdim::ObsDimension)::Int
    _check_nobs(tup, obsdim)
    length(tup) == 0 ? 0 : nobs(tup[1], obsdim)
end

function nobs(tup::Tuple, obsdims::Tuple)::Int
    _check_nobs(tup, obsdims)
    length(tup) == 0 ? 0 : nobs(tup[1], obsdims[1])
end

function getobs(tup::Tuple, obsdim::Tuple)
    _check_nobs(tup)
    map(getobs, tup, obsdim)
end

function getobs(tup::Tuple, obsdim::ObsDimension)
    _check_nobs(tup, obsdim)
    map(data -> getobs(data, obsdim), tup)
end

function getobs(tup::Tuple, indices)
    _check_nobs(tup)
    map(data -> getobs(data, indices), tup)
end

function getobs(tup::Tuple, indices, obsdim::ObsDimension)
    _check_nobs(tup, obsdim)
    map(data -> getobs(data, indices, obsdim), tup)
end

@generated function getobs(tup::Tuple, indices, obsdims::Tuple)
    N = length(obsdims.types)
    quote
        _check_nobs(tup, obsdims)
        # This line generates a tuple of N elements:
        # (getobs(tup[1], indices, obsdims[1]), getobs(tup[2], indi...
        $(Expr(:tuple, (:(getobs(tup[$i], indices, obsdims[$i])) for i in 1:N)...))
    end
end

_getobs_error() = throw(DimensionMismatch("the first argument (tuple with the buffers) must have the same length as the second argument (tuple with the data container)"))

@generated function getobs!(buffer::Tuple, tup::Tuple)
    N = length(buffer.types)
    N == length(tup.types) || _getobs_error()
    quote
        # _check_nobs(tup) # don't check because of single obs
        $(Expr(:tuple, (:(getobs!(buffer[$i],tup[$i])) for i in 1:N)...))
    end
end

@generated function getobs!(buffer::Tuple, tup::Tuple, indices, obsdim)
    N = length(buffer.types)
    N == length(tup.types) || _getobs_error()
    expr = if obsdim <: ObsDimension
        Expr(:tuple, (:(getobs!(buffer[$i], tup[$i], indices, obsdim)) for i in 1:N)...)
    else
        Expr(:tuple, (:(getobs!(buffer[$i], tup[$i], indices, obsdim[$i])) for i in 1:N)...)
    end
    quote
        # _check_nobs(tup, obsdim) # don't check because of single obs
        $expr
    end
end
