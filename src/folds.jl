"""
    FoldsView(data, train_indices, val_indices, [obsdim])

Description
============

Create a vector-like representation of `data` where each
individual element is partition of `data` in the form of a tuple
of two data subsets (a training- and a validation subset).

The purpose of `FoldsView` is to apply a precomputed sequence of
subset assignment indices to some data container in a convenient
manner. By itself, `FoldsView` is agnostic to any particular
repartitioning strategy (such as k-folds). Instead, the
assignments, `train_indices` and `val_indices`, need to be
precomputed by such a strategy and then passed to `FoldsView`
with a concrete `data` container. The resulting object can then
be queried for its individual folds using `getindex`, or simply
iterated over.

Arguments
==========

- **`data`** : The object describing the dataset. Can be of any
    type as long as it implements [`getobs`](@ref) and
    [`nobs`](@ref) (see Details for more information).

- **`train_indices`** : Vector of integer vectors containing the
    sequence of training assignments. This means that each
    element is a vector of indices that describe each *training*
    subset. The length of this vector must match `val_indices`.

- **`val_indices`** : Vector of integer vectors containing the
    sequence of validation assignments. This means that each
    element is a vector of indices that describe each
    *validation* subset. The length of this vector must match
    `train_indices`.

- **`obsdim`** : Optional. If it makes sense for the type of
    `data`, `obsdim` can be used to specify which dimension of
    `data` denotes the observations. It can be specified in a
    type-stable manner as a positional argument (see
    `?LearnBase.ObsDim`), or more conveniently as a smart keyword
    argument.

Details
========

For `FoldsView` to work on some data structure, the desired type
`MyType` must implement the following interface:

- `LearnBase.getobs(data::MyType, idx, [obsdim::ObsDimension])` :
    Should return the observation(s) indexed by `idx`.
    In what form is up to the user.
    Note that `idx` can be of type `Int` or `AbstractVector`.

- `StatsBase.nobs(data::MyType, [obsdim::ObsDimension])` :
    Should return the total number of observations in `data`

Author(s)
==========

- Christof Stocker (Github: https://github.com/Evizero)
- Tom Breloff (Github: https://github.com/tbreloff)

Examples
=========

    # Load iris data for demonstration purposes
    X, y = MLDataUtils.load_iris()

    # Compute train- and validation-partition indices using kfolds
    train_idx, val_idx = kfolds(nobs(X), 10)

    # These two vectors contain the indices vector for each partitioning
    @assert typeof(train_idx) <: Vector{Vector{Int64}}
    @assert typeof(val_idx)   <: Vector{UnitRange{Int64}}
    @assert length(train_idx) == length(val_idx) == 10

    # Using the repartitioning as an oterator
    for (train_X, val_X) in FoldsView(X, train_idx, val_idx)
        @assert size(train_X) == (4, 135)
        @assert size(val_X) == (4, 15)
    end

    # Calling kfolds with the dataset will create
    # the FoldsView for you automatically.
    # Thus this code is equivalent to above
    for (train_X, val_X) in kfolds(X, 10)
        @assert size(train_X) == (4, 135)
        @assert size(val_X) == (4, 15)
    end

    # leavout is a shortcut for setting k = nobs(X)
    for (train_X, val_X) in leaveout(X)
        @assert size(val_X) == (4, 1)
    end

see also
=========

[`kfolds`](@ref), [`leaveout`](@ref), [`splitobs`](@ref),
[`DataSubset`](@ref)
"""
struct FoldsView{T,D,O,A1<:AbstractArray,A2<:AbstractArray} <: DataView{T,D}
    data::D
    train_indices::A1
    val_indices::A2
    obsdim::O

    function FoldsView{T,D,O,A1,A2}(
            data::D, train_indices::A1, val_indices::A2, obsdim::O) where {T,D,O,A1<:AbstractArray,A2<:AbstractArray}
        n = nobs(data; obsdim = obsdim)
        (eltype(train_indices) <: AbstractArray{Int}) || throw(ArgumentError("The parameter \"train_indices\" must be an array of integer arrays"))
        (eltype(val_indices)  <: AbstractArray{Int}) || throw(ArgumentError("The parameter \"val_indices\" must be an array of integer arrays"))
        2 <= length(train_indices) <= n || throw(ArgumentError("The amount of train- and validation-indices must be within 2:$n respectively"))
        length(train_indices) == length(val_indices) || throw(DimensionMismatch("The amount of train- and validation-indices must match"))
        new{T,D,O,A1,A2}(data, train_indices, val_indices, obsdim)
    end
end

function FoldsView(data::D, train_indices::A1, val_indices::A2, obsdim::O) where {D,O,A1<:AbstractArray,A2<:AbstractArray}
    n = nobs(data; obsdim = obsdim)
    # TODO: Move this back into the inner constructor after the
    #       "T = typeof(...)" line below is removed
    (1 <= minimum(minimum.(train_indices)) && maximum(maximum.(train_indices)) <= n) || throw(DimensionMismatch("All training indices must be within 1:$n"))
    (1 <= minimum(minimum.(val_indices))  && maximum(maximum.(val_indices))  <= n) || throw(DimensionMismatch("All validation indices must be within 1:$n"))
    # FIXME: In 0.6 it should be possible to compute just the return
    #        type without executing the function
    T = typeof((datasubset(data, train_indices[1], obsdim), datasubset(data, val_indices[1])))
    FoldsView{T,D,O,A1,A2}(data, train_indices, val_indices, obsdim)
end

FoldsView(data, train_indices::AbstractArray, val_indices::AbstractArray; obsdim = default_obsdim(data)) =
    FoldsView(data, train_indices, val_indices, obsdim)

function FoldsView(data::T, train_indices::AbstractArray, val_indices::AbstractArray, obsdim) where T<:DataView
    @assert obsdim == data.obsdim
    @warn string("Trying to nest a ", T.name, " into an FoldsView, which is not supported. Returning FoldsView(parent(_)) instead")
    FoldsView(parent(data), train_indices, val_indices, obsdim)
end

# compare if both FoldsViews describe the same folds of the same data
# we don't care how the indices are stored, just that they match
# in their order and values
function Base.:(==)(fv1::FoldsView,fv2::FoldsView)
    fv1.data == fv2.data &&
        all(all(i1==i2 for (i1,i2) in zip(I1,I2)) for (I1,I2) in zip(fv1.train_indices,fv2.train_indices)) &&
        all(all(i1==i2 for (i1,i2) in zip(I1,I2)) for (I1,I2) in zip(fv1.val_indices,fv2.val_indices)) &&
        fv1.obsdim == fv2.obsdim
end

StatsBase.nobs(iter::FoldsView) = nobs(iter.data; obsdim = iter.obsdim)
Base.parent(iter::FoldsView) = iter.data
Base.length(iter::FoldsView) = length(iter.train_indices)

function Base.getindex(iter::FoldsView, i::Int)
    (datasubset(iter.data, iter.train_indices[i]),
     datasubset(iter.data, iter.val_indices[i]))
end

function Base.getindex(iter::FoldsView, i::AbstractVector)
    FoldsView(iter.data, iter.train_indices[i], iter.val_indices[i], iter.obsdim)
end

# compatibility with nested functions
default_obsdim(iter::FoldsView) = iter.obsdim

function Base.summary(A::FoldsView)
    string(length(A), "-fold FoldsView of ", nobs(A), " observations")
end

_datastr(data) = summary(data)
_datastr(data::Tuple) = string("(", join(map(_datastr, data), ", "), ")")
function Base.show(io::IO, ::MIME"text/plain", A::FoldsView)
    println(io, summary(A), ":")
    println(io, "  data: ", _datastr(A.data))
    # TODO: consider the case where fold sizes differ
    ntrain = length(first(A.train_indices))
    println(io, "  training: ", ntrain, " observations/fold")
    nval = length(first(A.val_indices))
    println(io, "  validation: ", nval, " observations/fold")
    print(io, "  obsdim: ", obsdim_string(A.obsdim))
end

"""
    kfolds(n::Integer, [k = 5]) -> Tuple

Compute the train/validation assignments for `k` repartitions of
`n` observations, and return them in the form of two vectors. The
first vector contains the index-vectors for the training subsets,
and the second vector the index-vectors for the validation subsets
respectively. A general rule of thumb is to use either `k = 5` or
`k = 10`. The following code snippet generates the indices
assignments for `k = 5`

```julia
julia> train_idx, val_idx = kfolds(10, 5);
```

Each observation is assigned to the validation subset once (and
only once). Thus, a union over all validation index-vectors
reproduces the full range `1:n`. Note that there is no random
assignment of observations to subsets, which means that adjacent
observations are likely to be part of the same validation subset.

```julia
julia> train_idx
5-element Array{Array{Int64,1},1}:
 [3,4,5,6,7,8,9,10]
 [1,2,5,6,7,8,9,10]
 [1,2,3,4,7,8,9,10]
 [1,2,3,4,5,6,9,10]
 [1,2,3,4,5,6,7,8]

julia> val_idx
5-element Array{UnitRange{Int64},1}:
 1:2
 3:4
 5:6
 7:8
 9:10
```
"""
function kfolds(n::Integer, k::Integer = 5)
    2 <= k <= n || throw(ArgumentError("n must be positive and k must to be within 2:$(max(2,n))"))
    # Compute the size of each fold. This is important because
    # in general the number of total observations might not be
    # divideable by k. In such cases it is custom that the remaining
    # observations are divided among the folds. Thus some folds
    # have one more observation than others.
    sizes = fill(floor(Int, n/k), k)
    for i = 1:(n % k)
        sizes[i] = sizes[i] + 1
    end
    # Compute start offset for each fold
    offsets = cumsum(sizes) .- sizes .+ 1
    # Compute the validation indices using the offsets and sizes
    val_indices = map((o,s)->(o:o+s-1), offsets, sizes)
    # The train indices are then the indicies not in validation
    train_indices = map(idx->setdiff(1:n,idx), val_indices)
    # We return a tuple of arrays
    train_indices, val_indices
end

"""
    kfolds(data, [k = 5], [obsdim]) -> FoldsView

Repartition a `data` container `k` times using a `k` folds
strategy and return the sequence of folds as a lazy
[`FoldsView`](@ref). The resulting `FoldsView` can then be
indexed into or iterated over. Either way, only data subsets are
created. That means that no actual data is copied until
[`getobs`](@ref) is invoked.

Conceptually, a k-folds repartitioning strategy divides the given
`data` into `k` roughly equal-sized parts. Each part will serve
as validation set once, while the remaining parts are used for
training. This results in `k` different partitions of `data`.

In the case that the size of the dataset is not dividable by the
specified `k`, the remaining observations will be evenly
distributed among the parts.

```julia
for (x_train, x_val) in kfolds(X, k = 10)
    # code called 10 times
    # nobs(x_val) may differ up to ±1 over iterations
end
```

Multiple variables are supported (e.g. for labeled data)

```julia
for ((x_train, y_train), val) in kfolds((X, Y), k = 10)
    # ...
end
```

By default the folds are created using static splits. Use
[`shuffleobs`](@ref) to randomly assign observations to the
folds.

```julia
for (x_train, x_val) in kfolds(shuffleobs(X), k = 10)
    # ...
end
```

see [`FoldsView`](@ref) for more info, or [`leaveout`](@ref) for
a related function.
"""
function kfolds(data; k::Integer = 5, obsdim = default_obsdim(data))
    n = nobs(data; obsdim = obsdim)
    train_indices, val_indices = kfolds(n, k)
    FoldsView(data, train_indices, val_indices, obsdim)
end

"""
    leaveout(n::Integer, [size = 1]) -> Tuple

Compute the train/validation assignments for `k ≈ n/size`
repartitions of `n` observations, and return them in the form of
two vectors. The first vector contains the index-vectors for the
training subsets, and the second vector the index-vectors for the
validation subsets respectively. Each validation subset will have
either `size` or `size+1` observations assigned to it. The
following code snippet generates the index-vectors for `size = 2`.

```julia
julia> train_idx, val_idx = leaveout(10, 2);
```

Each observation is assigned to the validation subset once (and
only once). Thus, a union over all validation index-vectors
reproduces the full range `1:n`. Note that there is no random
assignment of observations to subsets, which means that adjacent
observations are likely to be part of the same validation subset.

```julia
julia> train_idx
5-element Array{Array{Int64,1},1}:
 [3,4,5,6,7,8,9,10]
 [1,2,5,6,7,8,9,10]
 [1,2,3,4,7,8,9,10]
 [1,2,3,4,5,6,9,10]
 [1,2,3,4,5,6,7,8]

julia> val_idx
5-element Array{UnitRange{Int64},1}:
 1:2
 3:4
 5:6
 7:8
 9:10
```
"""
function leaveout(n::Integer, size::Integer = 1)
    1 <= size <= floor(n/2) || throw(ArgumentError("size must to be within 1:$(floor(Int,n/2))"))
    k = floor(Int, n / size)
    kfolds(n, k)
end

"""
    leaveout(data, [size = 1], [obsdim]) -> FoldsView

Repartition a `data` container using a k-fold strategy, where `k`
is chosen in such a way, that each validation subset of the
resulting folds contains roughly `size` observations. Defaults to
`size = 1`, which is also known as "leave-one-out" partitioning.

The resulting sequence of folds is returned as a lazy
[`FoldsView`](@ref), which can be index into or iterated over.
Either way, only data subsets are created. That means no actual
data is copied until [`getobs`](@ref) is invoked.

```julia
for (train, val) in leaveout(X, size = 2)
    # if nobs(X) is dividable by 2,
    # then nobs(val) will be 2 for each iteraton,
    # otherwise it may be 3 for the first few iterations.
end
```

see [`FoldsView`](@ref) for more info, or [`kfolds`](@ref) for a
related function.
"""
function leaveout(data; size = 1, obsdim = default_obsdim(data))
    n = nobs(data, obsdim)
    train_indices, val_indices = leaveout(n, size)
    FoldsView(data, train_indices, val_indices, obsdim)
end
