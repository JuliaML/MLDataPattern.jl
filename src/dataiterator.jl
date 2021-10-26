_length(iter) = _length(iter, Base.IteratorSize(iter))
_length(iter, ::Base.HasLength) = length(iter)
_length(iter, ::Base.HasShape)  = length(iter)
_length(iter, ::Base.IsInfinite) = Inf
_length(iter, ::Base.SizeUnknown) = "NA"

_length_str(iter) = _length_str(iter, Base.IteratorSize(iter))
_length_str(iter, ::Base.HasLength) = string(length(iter), ", ")
_length_str(iter, ::Base.HasShape)  = string(length(iter), ", ")
_length_str(iter, ::Base.IsInfinite) = ""
_length_str(iter, ::Base.SizeUnknown) = ""

# maybe I should use AbstractDataIterator here?
#Base.IteratorEltype(::Type{T}) where {T<:DataIterator} = Base.HasEltype()
#Base.eltype(::Type{DataIterator{E,T}}) where {E,T} = E

# There is no contract that says these methods will work
# It may be that some DataIterator subtypes do not support getindex
# and/or don't support collect
LearnBase.getobs(A::AbstractDataIterator) = map(getobs,collect(A))
LearnBase.getobs(A::AbstractDataIterator, i) = getobs(A[i])

DataSubset(data::T, indices) where {T<:AbstractDataIterator} =
    throw(MethodError(DataSubset, (data,indices)))

# To avoid overflow when infinite
_next_idx(iter, idx) = _next_idx(Base.IteratorSize(iter), idx)
_next_idx(::Base.IteratorSize, idx) = idx + 1
_next_idx(::Base.IsInfinite, idx) = 1

# --------------------------------------------------------------------

"""
    RandomObs(data, [count], [obsdim])

Description
============

Create an iterator that generates `count` randomly sampled
observations from `data`. In the case `count` is not provided,
it will generate random samples indefinitely.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`count`** : Optional. The number of randomly sampled
    observations that the iterator will generate before stopping.
    If omitted, the iterator will generate randomly sampled
    observations forever.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Details
========

For `RandomObs` to work on some data structure, the type of the
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

# go over 500 randomly sampled observations in X
i = 0
for x in RandomObs(X, 500) # also: RandomObs(X, count = 500)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert length(x) == 4
    i += 1
end
@assert i == 500

# if no count it provided the iterator will generate samples forever
for x in RandomObs(X)
    # this loop will never stop unless break is used
    if true; break; end
end

# also works for multiple data arguments (e.g. labeled data)
for (x,y) in RandomObs((X,Y), count = 100)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert typeof(y) <: String
end
```

see also
=========

[`BalancedObs`](@ref), [`RandomBatches`](@ref),
[`ObsView`](@ref), [`BatchView`](@ref), [`shuffleobs`](@ref),
[`DataSubset`](@ref), [`BufferGetObs`](@ref)
"""
struct RandomObs{T,O,I}
    data::T
    count::Int
    obsdim::O
end

function RandomObs(data::T, obsdim::O) where {T,O}
    RandomObs{T,O,Base.IsInfinite}(data, 1337, obsdim)
end

RandomObs(data::T, count::Int; obsdim = default_obsdim(data)) where {T} =
    RandomObs(data, count, obsdim)

# convenience constructor.
RandomObs(data, ::Nothing, obsdim) = RandomObs(data, obsdim)
function RandomObs(data; count = nothing, obsdim = default_obsdim(data))
    RandomObs(data, count, convert(LearnBase.ObsDimension,obsdim))
end

function Base.iterate(iter::RandomObs, idx = 1)
    idx > _length(iter) && return nothing
    (datasubset(iter.data, rand(1:nobs(iter.data; obsdim = iter.obsdim))),
     _next_idx(iter,idx))
end

Base.IteratorSize(::Type{RandomObs{T,O,I}}) where {T,O,I} = I()
Base.length(iter::RandomObs{T,O,Base.HasLength}) where {T,O} = iter.count
StatsBase.nobs(iter::RandomObs) = nobs(iter.data; obsdim = iter.obsdim)

function Base.summary(iter::RandomObs)
    io = IOBuffer()
    print(io, typeof(iter).name.name, "(")
    Base.showarg(io, iter.data, false)
    print(io, ", ", _length_str(iter))
    print(io, replace(string(iter.obsdim), "LearnBase." => ""))
    print(io, ')')
    first(readlines(seek(io,0)))
end

# --------------------------------------------------------------------

"""
    BalancedObs([f], data, [count], [obsdim])

Description
============

Create an iterator that generates `count` randomly sampled
observations from `data`. In the case `count` is not provided,
it will generate random samples indefinitely.

In contrast to [`RandomObs`](@ref), `BalancedObs` expects `data`
to be a labeled data container. It uses the label distribution of
`data` to make sure every label has an equal probability to be
sampled from.

Arguments
==========

- **`f`** : Optional. A function that should be applied to each
    observation individually in order to extract or compute the
    target for that observation. This function is only used once
    during construction to determine which label each observation
    belongs to.

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref),
    [`nobs`](@ref), and optionally [`gettargets`](@ref) (see
    Details for more information).

- **`count`** : Optional. The number of randomly sampled
    observations that the iterator will generate before stopping.
    If omitted, the iterator will generate randomly sampled
    observations forever.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Details
========

For `BalancedObs` to work on some data structure, the type of the
given variable `data` must implement the labeled data container
interface. See `?DataSubset` for more info.

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)

Examples
=========

```julia
# load first 55 observations of the iris data as example
# - 50 observations for "setosa"
# -  5 observations for "versicolor"
X, Y = MLDataUtils.load_iris(55)

# go over 100 balanced samples observations in X
num_versicolor = 0
for (x,y) in BalancedObs((X,Y), 100) # also: BalancedObs((X,Y), count = 100)
    @assert typeof(x) <: SubArray{Float64,1}
    @assert length(x) == 4
    # count how many times we sample a versicolor observation
    num_versicolor += y[] == "versicolor" ? 1 : 0
end
println(num_versicolor) # around 50

# if no count it provided the iterator will generate samples forever
for (x,y) in BalancedObs((X,Y))
    # this loop will never stop unless break is used
    if true; break; end
end
```

see also
=========

[`RandomObs`](@ref), [`targets`](@ref), [`RandomBatches`](@ref),
[`ObsView`](@ref), [`BatchView`](@ref), [`shuffleobs`](@ref),
[`DataSubset`](@ref), [`BufferGetObs`](@ref)
"""
struct BalancedObs{T,L,O,I} <: AbstractDataIterator
    data::T
    labelmap::L
    count::Int
    obsdim::O
end

function BalancedObs(f::Function, data, count::Int, obsdim)
    count > 0 || throw(ArgumentError("count has to be greater than 0"))
    lm = labelmap(eachtarget(f, data; obsdim = obsdim))
    BalancedObs{typeof(data),typeof(lm),typeof(obsdim),Base.HasLength}(data, lm, count, obsdim)
end
BalancedObs(data, count::Int, obsdim) = BalancedObs(identity, data, count, obsdim)

function BalancedObs(f::Function, data, obsdim)
    lm = labelmap(eachtarget(f, data; obsdim = obsdim))
    BalancedObs{typeof(data),typeof(lm),typeof(obsdim),Base.IsInfinite}(data, lm, 1337, obsdim)
end
BalancedObs(data, obsdim) = BalancedObs(identity, data, obsdim)

BalancedObs(data, count::Int; obsdim = default_obsdim(data)) =
    BalancedObs(data, count, obsdim)
BalancedObs(f::Function, data, count::Int; obsdim = default_obsdim(data)) =
    BalancedObs(f, data, count, obsdim)

# convenience constructor.
BalancedObs(data, ::Nothing, obsdim) = BalancedObs(data, obsdim)
BalancedObs(f::Function, data, ::Nothing, obsdim) = BalancedObs(f, data, obsdim)
function BalancedObs(data; count = nothing, obsdim = default_obsdim(data))
    BalancedObs(data, count, obsdim)
end
function BalancedObs(f::Function, data; count = nothing, obsdim = default_obsdim(data))
    BalancedObs(f, data, count, obsdim)
end

function Base.iterate(iter::BalancedObs, idx = 1)
    idx > _length(iter) && return nothing
    # uniformly random select a label
    obsidx = rand(iter.labelmap).second
    # uniformly random select observation from that label
    (datasubset(iter.data, rand(obsidx)),
     _next_idx(iter,idx))
end

Base.IteratorSize(::Type{BalancedObs{T,L,O,I}}) where {T,L,O,I} = I()
Base.length(iter::BalancedObs{T,L,O,Base.HasLength}) where {T,L,O} = iter.count
StatsBase.nobs(iter::BalancedObs) = nobs(iter.data; obsdim = iter.obsdim)

function Base.summary(iter::BalancedObs)
    io = IOBuffer()
    print(io, typeof(iter).name.name, '(')
    Base.showarg(io, iter.data, false)
    print(io, ", ", _length_str(iter))
    print(io, replace(string(iter.obsdim), "LearnBase." => ""))
    print(io, ')')
    first(readlines(seek(io,0)))
end

# --------------------------------------------------------------------

"""
    RandomBatches(data, [size], [count], [obsdim])

Description
============

Create an iterator that generates `count` randomly sampled
batches from `data` with a batch-size of `size` .
In the case `count` is not provided, it will generate random
batches indefinitely.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`size`** : Optional. The batch-size of each batch.
    I.e. the number of randomly sampled observations in each batch

- **`count`** : Optional. The number of randomly sampled batches
    that the iterator will generate before stopping. If omitted,
    the iterator will generate randomly sampled batches forever.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Details
========

For `RandomBatches` to work on some data structure, the type of
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

# go over 500 randomly sampled batches of batchsize 10
i = 0
for x in RandomBatches(X, 10, 500) # also: RandomObs(X, size = 10, count = 500)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert size(x) == (4,10)
    i += 1
end
@assert i == 500

# if no count it provided the iterator will generate samples forever
for x in RandomBatches(X, 10)
    # this loop will never stop unless break is used
    if true; break; end
end

# also works for multiple data arguments (e.g. labeled data)
for (x,y) in RandomBatches((X,Y), 10, 500)
    @assert typeof(x) <: SubArray{Float64,2}
    @assert typeof(y) <: SubArray{String,1}
end
```

see also
=========

[`RandomObs`](@ref), [`BatchView`](@ref), [`ObsView`](@ref),
[`shuffleobs`](@ref), [`DataSubset`](@ref), [`BufferGetObs`](@ref)
"""
struct RandomBatches{T,O,I} <: AbstractDataIterator
    data::T
    size::Int
    count::Int
    obsdim::O
end

function RandomBatches(data::T, size::Int, count::Int, obsdim::O) where {T,O}
    size  > 0 || throw(ArgumentError("size has to be greater than 0"))
    count > 0 || throw(ArgumentError("count has to be greater than 0"))
    RandomBatches{T,O,Base.HasLength}(data, size, count, obsdim)
end

function RandomBatches(data::T, size::Int, obsdim::O) where {T,O}
    size > 0 || throw(ArgumentError("size has to be greater than 0"))
    RandomBatches{T,O,Base.IsInfinite}(data, size, 1337, obsdim)
end

RandomBatches(data::T, size::Int, obsdim = default_obsdim(data)) where {T} =
    RandomBatches(data, size, obsdim)

RandomBatches(data::T, size::Int, count::Int, obsdim = default_obsdim(data)) where {T} =
    RandomBatches(data, size, count, obsdim)

# convenience constructor.
RandomBatches(data::T, size::Int, ::Nothing, obsdim) where {T} =
    RandomBatches(data, size, obsdim)

function RandomBatches(data::T; size::Int = -1, count = nothing, obsdim = default_obsdim(data)) where T
    nsize = size < 0 ? default_batch_size(data, obsdim)::Int : size
    RandomBatches(data, nsize, count, obsdim)
end

function Base.iterate(iter::RandomBatches, idx = 1)
    idx > _length(iter) && return nothing
    # maybe use StatsBase.sample instead of rand in order to avoid
    # replacement. That said I would like to avoid keyword arguments
    # and currently sample needs "replace" to be specified as such
    indices = rand(1:nobs(iter.data, iter.obsdim), iter.size)
    (datasubset(iter.data, indices), _next_idx(iter, idx))
end

Base.IteratorSize(::Type{RandomBatches{T,O,I}}) where {T,O,I} = I()
Base.length(iter::RandomBatches{T,O,Base.HasLength}) where {T,O} = iter.count
StatsBase.nobs(iter::RandomBatches) = nobs(iter.data, iter.obsdim)
batchsize(iter::RandomBatches) = iter.size

function Base.summary(iter::RandomBatches)
    io = IOBuffer()
    print(io, typeof(iter).name.name, "(")
    Base.showarg(io, iter.data, false)
    print(io, ", ")
    print(io, iter.size, ", ")
    print(io, _length_str(iter))
    print(io, replace(string(iter.obsdim), "LearnBase." => ""))
    print(io, ')')
    first(readlines(seek(io,0)))
end

# --------------------------------------------------------------------

"""
    BufferGetObs(iterator, [buffer])

A stateful iterator that stores the output of
`iterate(iterator,state)` into `buffer` using `getobs!(buffer,
...)`. Depending on the type of data provided by `iterator` this
may be more memory efficient than `getobs(...)`. In the case of
array data, for example, this allows for cache-efficient
processing of each element without allocating a temporary array.

Note that not all types of data support buffering, because it is
the author's choice to opt-in and implement a custom `getobs!`.
For those types that do not provide a custom `getobs!`, the
buffer will be ignored and the result of `getobs(...)` returned.

see [`eachobs`](@ref) and [`eachbatch`](@ref) for concrete examples.
"""
struct BufferGetObs{TElem,TIter}
    iter::TIter
    buffer::TElem
end

function BufferGetObs(iter::T) where T
    buffer = _getobs_eltype(iter, eltype(T))
    BufferGetObs{typeof(buffer),T}(iter, buffer)
end

_getobs_eltype(iter, ::Type) = getobs(first(iter))
_getobs_eltype(iter, ::Type{T}) where {T<:Tuple} = map(getobs, first(iter))

function Base.show(io::IO, iter::BufferGetObs{E,T}) where {E,T}
    if get(io, :compact, false)
        print(io, typeof(iter).name.name, "{", E, ",", T, "} with " , _length(iter), " elements")
    else
        print(io, summary(iter), "\n Iterator providing ", _length(iter), " elements")
    end
end

# TODO: simplify this mess
function Base.iterate(b::BufferGetObs)
    inner = Base.iterate(b.iter)
    inner == nothing && return nothing
    subset, nidx = inner
    (getobs!(b.buffer, subset), nidx)
end
function Base.iterate(b::BufferGetObs{T}) where T<:Tuple
    inner = Base.iterate(b.iter)
    inner == nothing && return nothing
    subset, nidx = inner
    (map(getobs!, b.buffer, subset), nidx)
end
function Base.iterate(b::BufferGetObs, idx)
    inner = Base.iterate(b.iter, idx)
    inner == nothing && return nothing
    subset, nidx = inner
    (getobs!(b.buffer, subset), nidx)
end
function Base.iterate(b::BufferGetObs{T}, idx) where T<:Tuple
    inner = Base.iterate(b.iter, idx)
    inner == nothing && return nothing
    subset, nidx = inner
    (map(getobs!, b.buffer, subset), nidx)
end

Base.eltype(::Type{BufferGetObs{E,T}}) where {E,T} = E
Base.IteratorSize(::Type{BufferGetObs{E,T}}) where {E,T} = Base.IteratorSize(T)
Base.length(b::BufferGetObs) = length(b.iter)
Base.size(b::BufferGetObs, I...) = size(b.iter, I...)
StatsBase.nobs(b::BufferGetObs) = nobs(b.iter)
batchsize(b::BufferGetObs) = batchsize(b.iter)

function Base.summary(b::BufferGetObs)
    io = IOBuffer()
    print(io, typeof(b).name.name, "(")
    Base.showarg(io, b.iter, false)
    print(io, ", ")
    Base.showarg(io, b.buffer, false)
    print(io, ')')
    first(readlines(seek(io,0)))
end

# --------------------------------------------------------------------

"""
    eachobs(data, [obsdim])

Iterate over `data` one observation at a time. If supported by
the type of `data`, a buffer will be preallocated and reused for
memory efficiency.

IMPORTANT: Avoid using `collect`, because in general each
iteration could return the same object with mutated values.
If that behaviour is undesired use `obsview` instead.

```julia
X = rand(4,100)
for x in eachobs(X)
    # loop entered 100 times
    @assert typeof(x) <: Vector{Float64}
    @assert size(x) == (4,)
end
```

In the case of arrays it is assumed that the observations are
represented by the last array dimension. This can be overwritten.

```julia
# This time flip the dimensions of the matrix
X = rand(100,4)
A = eachobs(X, obsdim=1)
# The behaviour remains the same as before
@assert eltype(A) <: Array{Float64,1}
@assert length(A) == 100
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachobs((X,Y))
    # ...
end
```

Note that internally `eachobs(data, obsdim)` maps to
`BufferGetObs(obsview(data, obsdim))`.

```julia
@assert typeof(eachobs(X)) <: BufferGetObs
@assert typeof(eachobs(X).iter) <: ObsView
```

This means that the following code:

```julia
for obs in eachobs(data, obsdim)
    # ...
end
```

is roughly equivalent to:

```julia
obs = getobs(data, 1, obsdim) # use first element to preallocate buffer
for _ in obsview(data, obsdim)
    getobs!(obs, _) # reuse buffer each iteration
    # ...
end
```

see [`BufferGetObs`](@ref), [`obsview`](@ref), and
[`getobs!`](@ref) for more info. also see [`eachbatch`](@ref) for
a mini-batch version.
"""
eachobs(data; obsdim = default_obsdim(data)) = BufferGetObs(ObsView(data, obsdim))

# --------------------------------------------------------------------

"""
    eachbatch(data, [size], [count], [obsdim])

Iterate over `data` one batch at a time. If supported by the type
of `data`, a buffer will be preallocated and reused for memory
efficiency.

IMPORTANT: Avoid using `collect`, because in general each
iteration could return the same object with mutated values.
If that behaviour is undesired use `BatchView` instead.

The (constant) batch-size can be either provided directly using
`size` or indirectly using `count`, which derives `size` based on
`nobs`. In the case that the size of the dataset is not dividable
by the specified (or inferred) `size`, the remaining observations
will be ignored.

```julia
X = rand(4,150)
for x in eachbatch(X, size = 10) # or: eachbatch(X, count = 15)
    # loop entered 15 times
    @assert typeof(x) <: Matrix{Float64}
    @assert size(x) == (4,10)
end
```

In the case of arrays it is assumed that the observations are
represented by the last array dimension. This can be overwritten.

```julia
# This time flip the dimensions of the matrix
X = rand(150,4)
A = eachbatch(X, size = 10, obsdim = 1)
# The behaviour remains the same as before
@assert eltype(A) <: Array{Float64,2}
@assert length(A) == 15
```

Multiple variables are supported (e.g. for labeled data)

```julia
for (x,y) in eachbatch((X,Y))
    # ...
end
```

Note that internally `eachbatch(data, ...)` maps to
`BufferGetObs(batchview(data, ...))`.

```julia
@assert typeof(eachbatch(X)) <: BufferGetObs
@assert typeof(eachbatch(X).iter) <: BatchView
```

This means that the following code:

```julia
for batch in eachbatch(data, batchsize, obsdim)
    # ...
end
```

is roughly equivalent to:

```julia
batch = getobs(data, collect(1:batchsize), obsdim) # use first element to preallocate buffer
for _ in batchview(data, batchsize, obsdim)
    getobs!(batch, _) # reuse buffer each iteration
    # ...
end
```

see [`BufferGetObs`](@ref), [`batchview`](@ref), and
[`getobs!`](@ref) for more info. also see [`eachobs`](@ref) for a
single-observation version.
"""
function eachbatch(data; size = -1, maxsize = -1, count = -1, obsdim = default_obsdim(data))
    maxsize != -1 && size != -1 && throw(ArgumentError("Providing both \"size\" and \"maxsize\" is not supported"))
    if maxsize != -1
        # set upto to true in order to allow a flexible batch size
        BufferGetObs(BatchView(data, maxsize, count, obsdim, true))
    else
        # force given batch size
        BufferGetObs(BatchView(data, size, count, obsdim))
    end
end

eachbatch(data, obsdim) =
    BufferGetObs(BatchView(data, -1, -1, obsdim))

eachbatch(data, size::Int, obsdim = default_obsdim(data), upto::Bool = false) =
    BufferGetObs(BatchView(data, size, -1, obsdim, upto))

eachbatch(data, size::Int, count::Int, obsdim = default_obsdim(data), upto::Bool = false) =
    BufferGetObs(BatchView(data, size, count, obsdim, upto))
