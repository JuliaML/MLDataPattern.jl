"""
    nobs(data, [obsdim]) -> Int

Return the total number of observations contained in `data`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense for
the type of `data`. See `?LearnBase.ObsDim` for more information.
"""
StatsBase.nobs(data; obsdim = default_obsdim(data)) = nobs(data, obsdim)
StatsBase.nobs(data, obsdim::Nothing) = nobs(data)

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
can be specified in a type-stable manner as a positional argument
(see `?LearnBase.ObsDim`), or more conveniently as a smart
keyword argument.
"""

# --------------------------------------------------------------------
# Global keyword definitions

LearnBase.getobs(data, idx; obsdim = default_obsdim(data)) = getobs(data, idx, obsdim)

# --------------------------------------------------------------------
# Arrays

# StatsBase.nobs(A::AbstractArray; obsdim = default_obsdim(A)) = size(A, obsdim)
# StatsBase.nobs(A::AbstractArray{T,0}; obsdim = default_obsdim(A)) where {T} = 1

# LearnBase.getobs(A::Array, obsdim) = A
# LearnBase.getobs(A::AbstractSparseArray, obsdim) = A
# LearnBase.getobs(A::SubArray, obsdim) = copy(A)
# LearnBase.getobs(A::SubArray{T,0}, obsdim) where {T} = A[1]
# LearnBase.getobs(A::AbstractArray, obsdim) = collect(A)

LearnBase.getobs(A::AbstractArray, idx, obsdim) = selectdim(A, obsdim, idx)

"""
    getobs!(buffer, data, [idx], [obsdim]) -> buffer

Write the observation(s) from `data` that correspond to the given
index/indices in `idx` into `buffer`. Note that `idx` can be of
type `Int` or `AbstractVector`.

Unless explicitly implemented for `data` it defaults to returning
`getobs(data, idx, obsdim)` in which case `buffer` is ignored.

If it makes sense for the type of `data`, `obsdim` can be used to
specify which dimension of `data` denotes the observations. It
can be specified in a type-stable manner as a positional argument
(see `?LearnBase.ObsDim`), or more conveniently as a smart
keyword argument.
"""
LearnBase.getobs!(buffer, A::AbstractArray, idx, obsdim) =
    getobs!(buffer, datasubset(A, idx, obsdim))

# --------------------------------------------------------------------
# Tuples

_check_nobs_error() =
    throw(DimensionMismatch("All data containers must have the same number of observations."))

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
    length(tup) == length(obsdims) ||
        throw(DimensionMismatch("Number of elements in obsdim doesn't match data."))
    n1 = nobs(tup[1], obsdims[1])
    for i=2:length(tup)
        nobs(tup[i], obsdims[i]) != n1 && _check_nobs_error()
    end
end

function LearnBase.nobs(tup::Tuple, obsdim::Nothing)::Int
    _check_nobs(tup)
    return length(tup) == 0 ? 0 : nobs(tup[1])
end

function LearnBase.nobs(tup::Tuple, obsdim)::Int
    _check_nobs(tup, obsdim)
    return length(tup) == 0 ? 0 : nobs(tup[1], obsdim)
end

function LearnBase.getobs(tup::Tuple, indices, obsdim)
    _check_nobs(tup, obsdim)
    return map(data -> getobs(data, indices, obsdim), tup)
end

@generated function LearnBase.getobs(tup::Tuple, indices, obsdims::Tuple)
    N = length(obsdims.types)

    return quote
        _check_nobs(tup, obsdims)
        # This line generates a tuple of N elements:
        # (getobs(tup[1], indices, obsdims[1]), getobs(tup[2], indi...
        $(Expr(:tuple, (:(getobs(tup[$i], indices, obsdims[$i])) for i in 1:N)...))
    end
end

_getobs_tuple_error() =
    throw(DimensionMismatch("The first argument (tuple with the buffers) must have the same length as the second argument (tuple with the data containers)."))

@generated function LearnBase.getobs!(buffer::Tuple, tup::Tuple, indices, obsdim)
    N = length(buffer.types)
    N == length(tup.types) || _getobs_tuple_error()
    expr = if obsdim <: Tuple
        Expr(:tuple, (:(getobs!(buffer[$i], tup[$i], indices, obsdim[$i])) for i in 1:N)...)
    else
        Expr(:tuple, (:(getobs!(buffer[$i], tup[$i], indices, obsdim)) for i in 1:N)...)
    end

    return quote
        # _check_nobs(tup, obsdim) # don't check because of single obs
        $expr
    end
end
