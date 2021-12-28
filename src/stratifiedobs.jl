splitobs(lm::Dict{T,Vector{I}}; at = 0.7) where {T,I<:Integer} =
    _splitobs(lm, at)

function _splitobs(lm::Dict{T,Vector{I}}, at::AbstractFloat) where {T,I<:Integer}
    0 < at < 1 || throw(ArgumentError("the parameter \"at\" must be in interval (0, 1)"))
    n = mapreduce(length, +, values(lm))
    k = nlabel(lm)
    # preallocate the indices vectors
    idx1 = Vector{I}()
    idx2 = Vector{I}()
    # sizehint will save us a few heavy memory allocations
    # we specify "+ k" to deal with trailing observations when
    # the number of observations from a class isn't divideable
    # by "at" or "1-at"
    sizehint!(idx1, ceil(Int, n * at     + k))
    sizehint!(idx2, ceil(Int, n * (1-at) + k))
    # loop through all label indices
    for indices in values(lm)
        i1, i2 = splitobs(indices; at = at)
        append!(idx1, i1)
        append!(idx2, i2)
    end
    idx1, idx2
end

@generated function _splitobs(lm::Dict{T,Vector{I}}, at::NTuple{N,AbstractFloat}) where {T,I<:Integer,N}
    quote
        (all(map(_ispos, at)) && sum(at) < 1) || throw(ArgumentError("all elements in \"at\" must be positive and their sum must be smaller than 1"))
        n = mapreduce(length, +, values(lm))
        k = nlabel(lm)
        # preallocate the indices vectors
        @nexprs $(N+1) i -> idx_i = Vector{I}()
        # sizehint will save us a few heavy memory allocations
        # we specify "+ k" to deal with trailing observations when
        # the number of observations from a class isn't divideable
        # by "at" or "1-at"
        @nexprs $(N) i -> sizehint!(idx_i, ceil(Int, n*at[i] + k))
        sizehint!($(Symbol(:idx_, Symbol(N+1))), ceil(Int, n*(1-sum(at)) + k))
        # loop through all label indices
        for indices in values(lm)
            tup = splitobs(indices; at = at)
            @nexprs $(N+1) i -> append!(idx_i, tup[i])
        end
        # return a tuple of all indices vectors
        @ntuple $(N+1) idx
    end
end

"""
    stratifiedobs([f], data, [p = 0.7], [shuffle = true], [obsdim], [rng]) -> Tuple

Partition the `data` into multiple disjoint subsets proportional
to the value(s) of `p`. The observations are assignmed to a data
subset using stratified sampling without replacement. These
subsets are then returned as a `Tuple` of subsets, where the
first element contains the fraction of observations of `data`
that is specified by the first float in `p`.

For example, if `p` is a `Float64` itself, then the return-value
will be a tuple with two elements (i.e. subsets), in which the
first element contains the fraction of observations specified by
`p` and the second element contains the rest. In the following
code the first subset `train` will contain around 70% of the
observations and the second subset `test` the rest. The key
difference to [`splitobs`](@ref) is that the class distribution
in `y` will actively be preserved in `train` and `test`.

```julia
train, test = stratifiedobs(y, p = 0.7)
```

If `p` is a tuple of `Float64` then additional subsets will be
created. In this example `train` will contain about 50% of the
observations, `val` will contain around 30%, and `test` the
remaining 20%.

```julia
train, val, test = stratifiedobs(y, p = (0.5, 0.3))
```

It is also possible to call `stratifiedobs` with multiple data
arguments as tuple, which all must have the same number of total
observations. Note that if `data` is a tuple, then it will be
assumed that the last element of the tuple contains the targets.

```julia
train, test = stratifiedobs((X, y), p = 0.7)
(X_train,y_train), (X_test,y_test) = stratifiedobs((X, y), p = 0.7)
```

The optional parameter `shuffle` determines if the resulting data
subsets should be shuffled. If `false`, then the observations in
the subsets will be grouped together according to their labels.

```julia
julia> y = ["a", "b", "b", "b", "b", "a"] # 2 imbalanced classes
6-element Array{String,1}:
 "a"
 "b"
 "b"
 "b"
 "b"
 "a"

julia> train, test = stratifiedobs(y, p = 0.5, shuffle = false)
(String["b","b","a"],String["b","b","a"])
```

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?ObsDim` for more information.

```julia
# 2 imbalanced classes in one-of-k encoding
julia> X = [1 0; 1 0; 1 0; 1 0; 0 1; 0 1]
6×2 Array{Int64,2}:
 1  0
 1  0
 1  0
 1  0
 0  1
 0  1

julia> train, test = stratifiedobs(argmax, X, p = 0.5, obsdim = 1)
([1 0; 1 0; 0 1], [0 1; 1 0; 1 0])
```

The optional (keyword) parameter `rng` allows one to specify the
random number generator used for shuffling. This is useful when
reproducible results are desired. By default, uses the global RNG.
See `Random` in Julia's standard library for more info.

```jldoctest; setup = :(using MLDataPattern)
julia> using Random: MersenneTwister

julia> X = [1:6;]
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6

julia> y = [:a, :b, :b, :b, :b, :a]
6-element Array{Symbol,1}:
 :a
 :b
 :b
 :b
 :b
 :a

julia> train, test = stratifiedobs((x, y), rng=MersenneTwister(42))
(([5, 3, 1, 4], [:b, :b, :a, :b]), ([2, 6], [:b, :a]))
```

For this function to work, the type of `data` must implement
[`nobs`](@ref) and [`getobs`](@ref). For example, the following
code allows `stratifiedobs` to work on a `DataTable`.

```julia
# Make DataTables.jl work
LearnBase.getobs(data::DataTable, i) = data[i,:]
StatsBase.nobs(data::DataTable) = nrow(data)
```

You can use the parameter `f` to specify how to extract or
retrieve the targets from each observation of the given `data`.

```julia
julia> data = DataTable(Any[rand(6), rand(6), [:a,:b,:b,:b,:b,:a]], [:X1,:X2,:Y])
6×3 DataTables.DataTable
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.226582  │ 0.0443222   │ a │
│ 2   │ 0.504629  │ 0.722906    │ b │
│ 3   │ 0.933372  │ 0.812814    │ b │
│ 4   │ 0.522172  │ 0.245457    │ b │
│ 5   │ 0.505208  │ 0.11202     │ b │
│ 6   │ 0.0997825 │ 0.000341996 │ a │

julia> train, test = stratifiedobs(row->row[:Y], data, 0.5);

julia> getobs(train)
3×3 DataTables.DataTable
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.933372  │ 0.812814    │ b │
│ 2   │ 0.522172  │ 0.245457    │ b │
│ 3   │ 0.0997825 │ 0.000341996 │ a │

julia> getobs(test)
3×3 DataTables.DataTable
│ Row │ X1       │ X2        │ Y │
├─────┼──────────┼───────────┼───┤
│ 1   │ 0.504629 │ 0.722906  │ b │
│ 2   │ 0.226582 │ 0.0443222 │ a │
│ 3   │ 0.505208 │ 0.11202   │ b │
```

see [`DataSubset`](@ref) for more information on data subsets.

see also [`undersample`](@ref), [`oversample`](@ref), [`splitobs`](@ref).
"""
function stratifiedobs(data; p = 0.7, shuffle = true, obsdim = default_obsdim(data), rng = Random.GLOBAL_RNG)
    _stratifiedobs(identity, data, p, shuffle, obsdim, rng)
end

function stratifiedobs(f, data; p = 0.7, shuffle = true, obsdim = default_obsdim(data), rng = Random.GLOBAL_RNG)
    _stratifiedobs(f, data, p, shuffle, obsdim, rng)
end

# function _stratifiedobs(data, p::AbstractFloat, args...)
#     _stratifiedobs(identity, data, p, args...)
# end

# function _stratifiedobs(data, p::NTuple{N,AbstractFloat}, args...) where N
#     _stratifiedobs(identity, data, p, args...)
# end

function _stratifiedobs(f, data, p::Union{NTuple,AbstractFloat}, shuffle::Bool, obsdim, rng::AbstractRNG)
    # The given data is always shuffled to qualify as performing
    # stratified sampling without replacement.
    data_shuf = shuffleobs(data; obsdim = obsdim, rng = rng)
    # FIXME: if i put the following line before `data_shuf` is
    #        defined, it breaks type inference for `data_shuf`.
    allowcontainer(stratifiedobs, data) || throw(MethodError(stratifiedobs, (f, data, shuffle, obsdim)))
    idx_tup = splitobs(labelmap(eachtarget(f, data_shuf, obsdim)); at = p)
    # Setting the parameter "shuffle = false" specifies that the
    # classes are ordered in the resulting subsets respectively.
    shuffle && foreach(x -> isempty(x) || shuffle!(rng, x), idx_tup)
    map(idx -> datasubset(data_shuf, idx), idx_tup)
end
