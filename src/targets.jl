# The targets logic is in some ways a bit more complex than the
# getobs logic. The main reason for this is that we want to
# support a wide variety of data storage types, as well as
# both, interation-based data and index-based data.
#
# A package author has two ways to customize the logic behind
# "targets" for their own data container types:
#
#   1. implementing "gettargets" for the data container type,
#      which bypasses "getobs" entirely.
#
#   2. implementing "gettarget" for the observation type,
#      which is applied on the result of "getobs".
#
# Note that if the optional first parameter is passed to "targets",
# it will always trigger "getobs", since it is assumed that the
# function is applied to the actual observation, and not the storage.
# Furthermore the first parameter is applied to each observation
# individually and not to the data as a whole. In general this means
# that the return type changes drastically.
#
# julia> X = rand(2, 3)
# 2×3 Array{Float64,2}:
#  0.105307   0.58033   0.724643
#  0.0433558  0.116124  0.89431
#
# julia> y = [1 3 5; 2 4 6]
# 2×3 Array{Int64,2}:
#  1  3  5
#  2  4  6
#
# julia> targets((X,y))
# 2×3 Array{Int64,2}:
#  1  3  5
#  2  4  6
#
# julia> targets(x->x, (X,y))
# 3-element Array{Array{Int64,1},1}:
#  [1,2]
#  [3,4]
#  [5,6]
#
# Here are two example scenarios that benefit from custom methods.
# The first one for "gettargets", and the second one for "gettarget".
#
# - Use-Case 1: Directory Based Image Source
#
#   Let's say you write a custom data storage that describes a
#   directory on your hard-drive. Each subdirectory contains a set
#   of large images that belong to a single class (the dir name).
#   This kind of data storage only loads the images itself if
#   they are actually needed (so on "getobs"). The targets however
#   are part of the metadata that is always loaded. So if we are
#   only interested in the targets (for example for resampling),
#   then we would like to avoid calling "getobs" if possible
#   Overloading "LearnBase.gettargets(::MyImageSource, i)" allows
#   a user to do just that. In other words it allows to provide
#   the targets of some observation(s) without ever calling "getobs".
#
# - Use-Case 2: DataFrames
#
#   DataFrames are a kind of data storage, where the targets are
#   as much part of the data as the features are (in contrast to
#   Use-Case 1). Here we are fine with "getobs" being called.
#   However, we still need to say which column actually describes
#   the features. We can do this by passing a function
#   "targets(row->row[1,:Y], dataframe)", or we can provide a
#   convenience syntax by overloading "gettarget".
#   "LearnBase.gettarget(col::Symbol, df::DataFrame) = df[1,col]"
#   This now allows us to call "targets(:Y, dataframe)".
#

# --------------------------------------------------------------------
# gettarget (singular)

"""
    gettarget([f], observation)

Use `f` (if provided) to extract the target from the single
`observation` and return it. It is used internally by
[`targets`](@ref) (only if `f` is provided) and by
[`eachtarget`](@ref) (always) on each individual observation.

```julia
julia> using DataFrames

julia> singleobs = DataFrame(X1=1.0, X2=0.5, Y=:a)
1×3 DataFrames.DataFrame
│ Row │ X1  │ X2  │ Y │
├─────┼─────┼─────┼───┤
│ 1   │ 1.0 │ 0.5 │ a │

julia> LearnBase.gettarget(x->x[1,:Y], singleobs)
:a
```

Even though this function is not exported, it is intended to be
extended by users to support their custom data storage types.
While not always necessary, it can make working with that storage
more convenient. The following example shows how to extend
`gettarget` for a more convenient use with a `DataFrame`. Note
that the first parameter is optional and need not be explicitly
supported.

```julia
julia> LearnBase.gettarget(col::Symbol, obs::DataFrame) = obs[1,col]

julia> LearnBase.gettarget(:Y, singleobs)
:a
```

By defining a custom `gettarget` method other functions (e.g.
[`targets`](@ref), [`eachtarget`](@ref), [`oversample`](@ref),
etc.) can make use of it as well. Note that these functions also
require [`nobs`](@ref) and [`getobs`](@ref) to be defined.

```julia
julia> LearnBase.getobs(data::DataFrame, i) = data[i,:]

julia> StatsBase.nobs(data::DataFrame) = nrow(data)

julia> data = DataFrame(X1=rand(3), X2=rand(3), Y=[:a,:b,:a])
3×3 DataFrames.DataFrame
│ Row │ X1       │ X2       │ Y │
├─────┼──────────┼──────────┼───┤
│ 1   │ 0.31435  │ 0.847857 │ a │
│ 2   │ 0.241307 │ 0.575785 │ b │
│ 3   │ 0.854685 │ 0.926744 │ a │

julia> targets(:Y, data)
3-element Array{Symbol,1}:
 :a
 :b
 :a
```
"""
gettarget

# custom "_" function to not recurse on tuples
# i.e. "_gettarget" interprets tuple while "gettarget" does not
# that means that "_gettarget((x,(y1,y2)))" --> "(y1,y2)", and NOT "y2"
# furthermore "_gettargets" assumes that it is called with a subset
# of one observation and will thus trigger "getobs"
@inline _gettarget(f, data) = gettarget(f, getobs_targetfun(data))

# no nobs check because this should be a single observation
@inline _gettarget(f, tup::NTuple{N,Any}) where {N} = gettarget(f, getobs_targetfun(tup[N]))

# gettarget is intended to be defined by the user
@inline gettarget(data) = data
@inline gettarget(f, data) = f(gettarget(data))

# if a target extraction function is used, then
# we don't need to call "getobs" on a "SubArray".
getobs_targetfun(data) = getobs(data)
getobs_targetfun(tup::Tuple) = map(getobs_targetfun, tup)
getobs_targetfun(A::AbstractArray) = A
getobs_targetfun(A::AbstractArray{T,0}) where {T} = A[1]

# --------------------------------------------------------------------
# gettargets (plural)

"""
    gettargets(data, [idx], [obsdim])

Return the targets corresponding to the observation-indices
`idx`. Note that `idx` can be of type `Int` or `AbstractVector`.

Implementing this function for a custom type of `data` is
optional. It is particularly useful if the targets in `data` can
be provided without invoking `getobs`. For example if you have a
remote data-source where the labels are part of some metadata
that is locally available.

If implemented, calling [`targets`](@ref) will invoke this
method, instead of [`gettarget`](@ref) on the result of
[`getobs`](@ref). This can make operations that require targets,
such as [`oversample`](@ref), much more efficient.

If it makes sense for the type of `data`, `obsdim` can be used to
specify which dimension of `data` denotes the observations. It
can be specified in a type-stable manner as a positional argument
(see `?ObsDim`), or more conveniently as a smart keyword
argument.
"""
gettargets

# custom "_" function to throw away Undefined obsdims.
# this is important so that a user can leave out obsdim when
# implementing a custom method that doesn't need it.
# @inline _gettargets(data, ::ObsDim.Undefined) = gettargets(data)
# @inline _gettargets(data, idx, ::ObsDim.Undefined) = gettargets(data, idx)
# @inline _gettargets(data) = gettargets(data)
# @inline _gettargets(data, args...) = gettargets(data, args...)

# We use this _gettargets_dispatch_idx function to avoid ambiguity
@inline gettargets(data, idx; kwargs...) =
    _gettargets_dispatch_idx(data, idx; kwargs...)

@inline _gettargets_dispatch_idx(data, idx; kwargs...) =
    gettargets(obsview(DataSubset(data, idx)))

@inline _gettargets_dispatch_idx(data, idx::Int; kwargs...) =
    gettarget(identity, getobs_targetfun(datasubset(data, idx)))

# This method prevents ObsView to happen by default for Arrays.
# Thus targets will return arrays in their original shape.
@inline gettargets(data::AbstractArray, idx; obsdim = default_obsdim(data)) =
    getobs(data, idx; obsdim = obsdim)

# DataSubset will query the underlying data using "gettargets"
# this way custom data storage types can provide the targets
# without having to trigger "getobs". Support optional
# @inline gettargets(subset::DataSubset) =
#     _gettargets(subset.data, subset.indices, subset.obsdim)

# @inline gettargets(subset::DataSubset, idx) =
#     _gettargets(subset.data, _view(subset.indices, idx), subset.obsdim)

# function gettargets(subset::DataSubset, obsdim::Union{Tuple, ObsDimension})
#     @assert obsdim === subset.obsdim
#     _gettargets(subset.data, subset.indices, subset.obsdim)
# end

function gettargets(subset::DataSubset, idx; obsdim = default_obsdim(subset))
    @assert obsdim == subset.obsdim
    gettargets(subset, idx)
end

gettargets(tup::Tuple) = map(gettargets, tup)

# --------------------------------------------------------------------
# targets (exported API)

"""
    targets([f], data, [obsdim])

Extract the concrete targets from `data` and return them.

This function is eager in the sense that it will always call
[`getobs`](@ref) unless a custom method for [`gettargets`](@ref)
is implemented for the type of `data`. This will make sure that
actual values are returned (in contrast to placeholders such as
`DataSubset` or `SubArray`).

```julia
julia> targets(DataSubset([1,2,3]))
3-element Array{Int64,1}:
 1
 2
 3
```

If `data` is a tuple, then the convention is that the last
element of the tuple contains the targets and the function is
recursed once (and only once).

```julia
julia> targets(([1,2], [3,4]))
2-element Array{Int64,1}:
 3
 4

julia> targets(([1,2], ([3,4], [5,6])))
([3,4],[5,6])
```

If `f` is provided, then [`gettarget`](@ref) will be applied to
each observation in `data` and the results will be returned as a
vector.

```julia
julia> targets(argmax, [1 0 1; 0 1 0])
3-element Array{Int64,1}:
 1
 2
 1
```

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?ObsDim` for more information.

```julia
julia> targets(argmax, [1 0; 0 1; 1 0], obsdim=1)
3-element Array{Int64,1}:
 1
 2
 1
```
"""
targets

# keyword based convenience API
# targets(data; obsdim=default_obsdim(data)) =
#     targets(identity, data, convert(LearnBase.ObsDimension,obsdim))

# targets(f, data; obsdim=default_obsdim(data)) =
#     targets(f, data, convert(LearnBase.ObsDimension,obsdim))

# default to identity function. This is later dispatched on
@inline targets(data; obsdim = default_obsdim(data)) =
    targets(identity, data; obsdim = obsdim)

# @inline targets(tup::Tuple, obsdim::Tuple) =
#     targets(identity, tup, obsdim)

# only dispatch on tuples once (nested tuples are not interpreted)
# here we swapped the naming convention because the exposed function
# should always be the one without an "_"
# i.e. "targets" interprets tuple, while "_targets" does not
# that means that "targets((X,(Y1,Y2)))" --> "(Y1,Y2)", and NOT "Y2"
targets(f, tup::NTuple{N, Any}; obsdim = default_obsdim(tup)) where N =
    _targets_tuple(f, tup, obsdim)


function _targets_tuple(f, tup::NTuple{N,Any}, obsdim) where N
    _check_nobs(tup, obsdim)
    _targets(f, tup[N], obsdim)
end
function _targets_tuple(f, tup::NTuple{N,Any}, obsdim::Tuple) where N
    _check_nobs(tup, obsdim)
    _targets(f, tup[N], obsdim[N])
end

# @inline targets(f, data, obsdim) = _targets(f, data, obsdim)

# Batch Views
targets(f, data::BatchView) =
    map(x->targets(f,x), data)

# Obs Views
targets(f, data::ObsView) =
    map(x->_gettarget(f,x), data)

# custom "_" function to not recurse on tuples
# Here we decide if "getobs" will be triggered based on "f"
@inline targets(::typeof(identity), data; obsdim = default_obsdim(data)) =
    gettargets(data, 1:nobs(data; obsdim = obsdim); obsdim = obsdim)

targets(f, data; obsdim = default_obsdim(data)) =
    map(x -> gettarget(f, getobs_targetfun(x)), obsview(data, obsdim))

# --------------------------------------------------------------------
# eachtarget (lazy version of targets for iterating)

# keyword based convenience API
# eachtarget(data; obsdim=default_obsdim(data)) =
#     eachtarget(identity, data, convert(LearnBase.ObsDimension,obsdim))

# eachtarget(f, data; obsdim=default_obsdim(data)) =
#     eachtarget(f, data, convert(LearnBase.ObsDimension,obsdim))

@inline eachtarget(data; obsdim = default_obsdim(data)) =
    eachtarget(identity, data, obsdim)

# @inline eachtarget(tup::Tuple, obsdim::Tuple) =
#     eachtarget(identity, tup, obsdim)

# go with _gettargets (avoids getobs)

eachtarget(f::typeof(identity), data; obsdim = default_obsdim(data)) =
    (gettargets(data, i; obsdim = obsdim) for i in 1:nobs(data; obsdim = obsdim))

eachtarget(f::typeof(identity), tup::NTuple{N, Any}; obsdim = default_obsdim(tup)) where N =
    _eachtarget_tuple(f, tup, obsdim)

function _eachtarget_tuple(::typeof(identity), tup::NTuple{N,Any}, obsdim) where N
    _check_nobs(tup, obsdim)
    (gettargets(tup[N], i; obsdim = obsdim) for i in 1:nobs(tup[N]; obsdim = obsdim))
end
function _eachtarget_tuple(::typeof(identity), tup::NTuple{N,Any}, obsdim::Tuple) where N
    _check_nobs(tup, obsdim)
    (gettargets(tup[N], i; obsdim = obsdim[N]) for i in 1:nobs(tup[N]; obsdim = obsdim[N]))
end

# go with _gettarget (triggers getobs)

@inline eachtarget(f, data; obsdim = default_obsdim(data)) =
    eachtarget(f, obsview(data; obsdim = obsdim))

function eachtarget(f, data::ObsView; obsdim = default_obsdim(data))
    @assert obsdim == default_obsdim(data)
    (_gettarget(f,x) for x in data)
end

# function eachtarget(f, data::AbstractObsView, obsdim=default_obsdim(data))
#     @assert obsdim === default_obsdim(data)
#     (_gettarget(f,x) for x in data)
# end
