"""
    splitobs(n::Int, [at = 0.7]) -> Tuple

Pre-compute the indices for two disjoint subsets and return them
as a tuple of two ranges. The first range will span the first
`at` fraction of possible indices, while the second range will
cover the rest. These indices are applicable to any data
container of size `n`.

```julia
julia> splitobs(100, at = 0.7)
# (1:70,71:100)
```
"""
splitobs(n::Int; at = 0.7) = splitobs(n, at)

# partition into 2 sets
function splitobs(n::Int, at::AbstractFloat)
    0 < at < 1 || throw(ArgumentError("the parameter \"at\" must be in interval (0, 1)"))
    n1 = clamp(round(Int, at*n), 1, n)
    (1:n1, n1+1:n)
end

# has to be outside the generated function
_ispos(x) = x > 0
# partition into length(at)+1 sets
# we use @generated because we compute "N+1"
@generated function splitobs(n::Int, at::NTuple{N,AbstractFloat}) where N
    quote
        (all(map(_ispos, at)) && sum(at) < 1) || throw(ArgumentError("all elements in \"at\" must be positive and their sum must be smaller than 1"))
        nleft = n
        lst = UnitRange{Int}[]
        for (i, sz) in enumerate(at)
            ni = clamp(round(Int, sz*n), 0, nleft)
            push!(lst, n-nleft+1:n-nleft+ni)
            nleft -= ni
        end
        push!(lst, n-nleft+1:n)
        $(Expr(:tuple, (:(lst[$i]) for i in 1:N+1)...))
    end
end

"""
    splitobs(data, [at = 0.7], [obsdim]) -> Tuple

Split the `data` into multiple subsets proportional to the
value(s) of `at`.

Note that this function will perform the splits statically and
thus not perform any randomization. The function creates a
`NTuple` of data subsets in which the first N-1 elements/subsets
contain the fraction of observations of `data` that is specified
by `at`.

For example, if `at` is a `Float64` then the return-value will be
a tuple with two elements (i.e. subsets), in which the first
element contains the fracion of observations specified by `at`
and the second element contains the rest. In the following code
the first subset `train` will contain the first 70% of the
observations and the second subset `test` the rest.

```julia
train, test = splitobs(X, at = 0.7)
```

If `at` is a tuple of `Float64` then additional subsets will be
created. In this example `train` will have the first 50% of the
observations, `val` will have next 30%, and `test` the last 20%

```julia
train, val, test = splitobs(X, at = (0.5, 0.3))
```

It is also possible to call `splitobs` with multiple data
arguments as tuple, which all must have the same number of total
observations. This is useful for labeled data.

```julia
train, test = splitobs((X, y), at = 0.7)
(x_train,y_train), (x_test,y_test) = splitobs((X, y), at = 0.7)
```

If the observations should be randomly assigned to a subset,
then you can combine the function with `shuffleobs`

```julia
# This time observations are randomly assigned.
train, test = splitobs(shuffleobs((X,y)), at = 0.7)
```

When working with arrays one may want to choose which dimension
represents the observations. By default the last dimension is
assumed, but this can be overwritten.

```julia
# Here we say each row represents an observation
train, test = splitobs(X, obsdim = 1)
```

The functions also provide a type-stable API

```julia
# By avoiding keyword arguments, the compiler can infer the return type
train, test = splitobs((X,y), 0.7)
train, test = splitobs((X,y), 0.7, ObsDim.First())
```

see [`DataSubset`](@ref) for more information on data subsets.

see [`stratifiedobs`](@ref) for a related function that preserves
the target distribution.
"""
splitobs(data; at = 0.7, obsdim = default_obsdim(data)) =
    splitobs(data, at, obsdim)

# partition into 2 sets
function splitobs(data, at::AbstractFloat, obsdim=default_obsdim(data))
    allowcontainer(splitobs, data) || throw(MethodError(splitobs, (data,at,obsdim)))
    n = nobs(data, obsdim)
    idx1, idx2 = splitobs(n, at)
    datasubset(data, idx1, obsdim), datasubset(data, idx2, obsdim)
end

# partition into length(at)+1 sets
function splitobs(data, at::NTuple{N,AbstractFloat}, obsdim=default_obsdim(data)) where N
    allowcontainer(splitobs, data) || throw(MethodError(splitobs, (data,at,obsdim)))
    n = nobs(data, obsdim)
    map(idx->datasubset(data, idx, obsdim), splitobs(n, at))
end
