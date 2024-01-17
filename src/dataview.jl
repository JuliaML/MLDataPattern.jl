Base.IndexStyle(::Type{T}) where {T<:DataView} = IndexLinear()
Base.size(A::DataView) = (length(A),)
Base.lastindex(A::DataView) = length(A)
getobs(A::DataView) = map(getobs, A)
getobs(A::DataView, i) = getobs(A[i])
getobs(A::DataView{T}) where {T<:Tuple} = map(i->getobs(A,i), 1:length(A))
getobs(A::DataView{T}, i::Integer) where {T<:Tuple} = map(getobs, A[i])

allowcontainer(fun, ::AbstractObsView) = true
allowcontainer(fun, ::DataView) = false

# for proper dispatch to trump the abstract arrays one
for T in (ObsDim.Constant, ObsDim.Last, Tuple)
    @eval function nobs(A::DataView, obsdim::$T)
        @assert obsdim === default_obsdim(A)
        nobs(A)
    end
    @eval function getobs(A::DataView, idx, obsdim::$T)
        @assert obsdim === default_obsdim(A)
        getobs(A, idx)
    end
end

# --------------------------------------------------------------------

"""
    ObsView(data, [obsdim])

Description
============

Create a view of the given `data` in the form of a vector of
individual observations. Any data access is delayed until
`getindex` is called, and even `getindex` returns the result of
[`datasubset`](@ref) which in general avoids data movement until
[`getobs`](@ref) is called.

If used as an iterator, the view will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
lazy subset to the current observation.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Methods
========

Aside from the `AbstractArray` interface following additional
methods are provided:

- **`getobs(data::ObsView, indices::AbstractVector)`** :
    Returns a `Vector` of indivdual observations specified by
    `indices`.

- **`nobs(data::ObsView)`** :
    Returns the number of observations in `data` that the
    iterator will go over.

Details
========

For `ObsView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See `?DataSubset` for more info.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
X, Y = MLDataUtils.load_iris()

A = obsview(X)
@assert typeof(A) <: ObsView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,1}
@assert length(A) == 150 # Iris has 150 observations
@assert size(A[1]) == (4,) # Iris has 4 features

for x in obsview(X)
    @assert typeof(x) <: SubArray{Float64,1}
end

# iterate over each individual labeled observation
for (x,y) in obsview((X,Y))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end

# same but in random order
for (x,y) in obsview(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

[`eachobs`](@ref), [`BatchView`](@ref), [`shuffleobs`](@ref),
[`getobs`](@ref), [`nobs`](@ref), [`DataSubset`](@ref)
"""
struct ObsView{TElem,TData,O} <: AbstractObsView{TElem,TData}
    data::TData
    obsdim::O
end

function ObsView(data::T, obsdim::O) where {T,O}
    E = typeof(datasubset(data, 1, obsdim))
    ObsView{E,T,O}(data,obsdim)
end

function ObsView(A::T, obsdim) where T<:DataView
    @assert obsdim == A.obsdim
    @warn string("Trying to nest a ", T.name, " into an ObsView, which is not supported. Returning ObsView(parent(_)) instead")
    ObsView(parent(A), obsdim)
end

function ObsView(A::ObsView, obsdim)
    @assert obsdim == A.obsdim
    A
end

ObsView(data; obsdim = default_obsdim(data)) =
    ObsView(data, convert(LearnBase.ObsDimension,obsdim))

nobs(A::ObsView) = nobs(A.data, A.obsdim)
Base.parent(A::ObsView) = A.data
Base.length(A::ObsView) = nobs(A)
Base.getindex(A::ObsView, i::Int) = datasubset(A.data, i, A.obsdim)
Base.getindex(A::ObsView, i::AbstractVector) =
    ObsView(datasubset(A.data, i, A.obsdim), A.obsdim)

# compatibility with nested functions
default_obsdim(A::ObsView) = A.obsdim

@doc (@doc ObsView)
const obsview = ObsView

function Base.showarg(io::IO, A::ObsView, toplevel)
    print(io, "obsview(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, replace(string(A.obsdim), "LearnBase." => ""))
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simpify
end

# --------------------------------------------------------------------

default_batch_size(source, obsdim) = clamp(div(nobs(source,obsdim), 5), 2, 100)

"""
Helper function to compute sensible and compatible values for the
`size` and `count`
"""
function _compute_batch_settings(source, size::Int = -1, count::Int = -1, obsdim = default_obsdim(source), upto = false)
    num_observations = nobs(source, obsdim)::Int
    @assert num_observations > 0
    if upto && size > num_observations
        size = num_observations
    end
    size  <= num_observations || throw(ArgumentError("Specified batch-size is too large for the given number of observations"))
    count <= num_observations || throw(ArgumentError("Specified batch-count is too large for the given number of observations"))
    if size > 0 && upto
        while num_observations % size != 0 && size > 1
            size = size - 1
        end
    end
    if size <= 0 && count <= 0
        # no batch settings specified, use default size and as many batches as possible
        size = default_batch_size(source, obsdim)::Int
        count = floor(Int, num_observations / size)
    elseif size <= 0
        # use count to determine size. try use all observations
        size = floor(Int, num_observations / count)
    elseif count <= 0
        # use size and as many batches as possible
        count = floor(Int, num_observations / size)
    else
        # use count just for boundscheck
        max_batchcount = floor(Int, num_observations / size)
        count <= max_batchcount || throw(ArgumentError("Specified number of partitions is too large for the specified size"))
    end

    # check if the settings will result in all data points being used
    unused = num_observations - size*count
    if unused > 0
        @warn "The specified values for size and/or count will result in $unused unused data points" maxlog=1
    end
    size::Int, count::Int
end

"""
Helper function to translate a batch-index into a range of observations.
"""
function _batchrange(batchsize::Int, batchindex::Int)
    startidx = (batchindex - 1) * batchsize + 1
    startidx:(startidx + batchsize - 1)
end

# --------------------------------------------------------------------

"""
    BatchView(data, [size|maxsize], [count], [obsdim])

Description
============

Create a view of the given `data` that represents it as a vector
of batches. Each batch will contain an equal amount of
observations in them. The number of batches and the batch-size
can be specified using (keyword) parameters `count` and `size`.
In the case that the size of the dataset is not dividable by the
specified (or inferred) `size`, the remaining observations will
be ignored with a warning.

Note that any data access is delayed until `getindex` is called,
and even `getindex` returns the result of [`datasubset`](@ref)
which in general avoids data movement until [`getobs`](@ref) is
called.

If used as an iterator, the object will iterate over the dataset
once, effectively denoting an epoch. Each iteration will return a
mini-batch of constant [`nobs`](@ref), which effectively allows
to iterator over [`data`](@ref) one batch at a time.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`size`** : The batch-size of each batch. I.e. the number of
    observations that each batch must contain.

- **`maxsize`** : The maximum batch-size of each batch. I.e. the
    number of observations that each batch should contain. If the
    number of total observation is not divideable by the size it
    will be reduced until it is.

- **`count`** : The number of batches that the view will contain.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Methods
========

Aside from the `AbstractArray` interface following additional
methods are provided.

- **`getobs(data::BatchView, batchindices)`** :
    Returns a `Vector` of the batches specified by `batchindices`.

- **`nobs(data::BatchView)`** :
    Returns the total number of observations in `data`. Note that
    unless the batch-size is 1, this number will differ from
    `length`.

Details
========

For `BatchView` to work on some data structure, the type of the
given variable `data` must implement the data container
interface. See `?DataSubset` for more info.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

```julia
using MLDataUtils
X, Y = MLDataUtils.load_iris()

A = batchview(X, size = 30)
@assert typeof(A) <: BatchView <: AbstractVector
@assert eltype(A) <: SubArray{Float64,2}
@assert length(A) == 5 # Iris has 150 observations
@assert size(A[1]) == (4,30) # Iris has 4 features

# 5 batches of size 30 observations
for x in batchview(X, size = 30)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert nobs(x) === 30
end

# 7 batches of size 20 observations
# Note that the iris dataset has 150 observations,
# which means that with a batchsize of 20, the last
# 10 observations will be ignored
for (x,y) in batchview((X,Y), size = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert nobs(x) === nobs(y) === 20
end

# 10 batches of size 15 observations
for (x,y) in batchview((X,Y), maxsize = 20)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert nobs(x) === nobs(y) === 15
end

# randomly assign observations to one and only one batch.
for (x,y) in batchview(shuffleobs((X,Y)))
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end

# iterate over the first 2 batches of 15 observation each
for (x,y) in batchview((X,Y), size=15, count=2)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
    @assert size(x) == (4, 15)
    @assert size(y) == (15,)
end
```

see also
=========

[`eachbatch`](@ref), [`ObsView`](@ref), [`shuffleobs`](@ref),
[`getobs`](@ref), [`nobs`](@ref), [`DataSubset`](@ref)
"""
struct BatchView{TElem,TData,O} <: AbstractBatchView{TElem,TData}
    data::TData
    size::Int
    count::Int
    obsdim::O
end

function BatchView(data::T, size::Int, count::Int, obsdim::O = default_obsdim(data), upto::Bool = false) where {T,O}
    nsize, ncount = _compute_batch_settings(data, size, count, obsdim, upto)
    E = typeof(datasubset(data, 1:nsize, obsdim))
    BatchView{E,T,O}(data, nsize, ncount, obsdim)
end

function BatchView(data::T, size::Int, obsdim::O = default_obsdim(data), upto::Bool = false) where {T,O<:Union{Tuple,ObsDimension}}
    BatchView(data, size, -1, obsdim, upto)
end

function BatchView(A::BatchView, size::Int, count::Int, obsdim, upto::Bool = false)
    @assert obsdim == A.obsdim
    BatchView(parent(A), size, count, obsdim, upto)
end

BatchView(data, obsdim::Union{Tuple,ObsDimension}) =
    BatchView(data, -1, -1, obsdim)

function BatchView(data; size = -1, maxsize = -1, count = -1, obsdim = default_obsdim(data))
    maxsize != -1 && size != -1 && throw(ArgumentError("Providing both \"size\" and \"maxsize\" is not supported"))
    if maxsize != -1
        # set upto to true in order to allow a flexible batch size
        BatchView(data, maxsize, count, convert(LearnBase.ObsDimension,obsdim), true)
    else
        # force given batch size
        BatchView(data, size, count, convert(LearnBase.ObsDimension,obsdim))
    end
end

allowcontainer(::typeof(shuffleobs), ::BatchView) = true
allowcontainer(::typeof(splitobs), ::BatchView) = true

"""
    batchsize(data) -> Int

Return the fixed size of each batch in `data`.
"""
batchsize(A::BatchView) = A.size
nobs(A::BatchView) = A.count * A.size
Base.parent(A::BatchView) = A.data
Base.length(A::BatchView) = A.count
Base.getindex(A::BatchView, batchindex::Int) =
    datasubset(A.data, _batchrange(A.size, batchindex), A.obsdim)

function Base.getindex(A::BatchView, batchindices::AbstractVector)
    obsindices = union((_batchrange(A.size, bi) for bi in batchindices)...)::Vector{Int}
    BatchView(datasubset(A.data, obsindices, A.obsdim), A.size, -1, A.obsdim)
end

# compatibility with nested functions
default_obsdim(A::BatchView) = A.obsdim

@doc (@doc BatchView)
const batchview = BatchView

function Base.showarg(io::IO, A::BatchView, toplevel)
    print(io, "batchview(")
    Base.showarg(io, parent(A), false)
    print(io, ", ")
    print(io, A.size, ", ")
    print(io, A.count, ", ")
    print(io, replace(string(A.obsdim), "LearnBase." => ""))
    print(io, ')')
    toplevel && print(io, " with eltype ", nameof(eltype(A))) # simpify
end

# --------------------------------------------------------------------

# if subsetting a DataView, then DataView the subset instead.
for fun in (:DataSubset, :datasubset), O in (ObsDimension, Tuple)
    @eval @generated function ($fun)(A::T, i, obsdim::$O) where T<:AbstractObsView
        quote
            @assert obsdim == A.obsdim
            ($(T.name.name))(($($fun))(parent(A), i, obsdim), obsdim)
        end
    end
    @eval function ($fun)(A::BatchView, i, obsdim::$O)
        @assert obsdim == A.obsdim
        length(i) < A.size && throw(ArgumentError("The chosen batch-size ($(A.size)) is greater than the number of observations ($(length(i)))"))
        BatchView(($fun)(parent(A), i, obsdim), A.size, -1, obsdim)
    end
end
