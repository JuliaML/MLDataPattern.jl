"""
    shuffleobs(data, [obsdim])

Return a "subset" of `data` that spans all observations, but
has the order of the observations shuffled.

The values of `data` itself are not copied. Instead only the
indices are shuffled. This function calls [`datasubset`](@ref) to
accomplish that, which means that the return value is likely of a
different type than `data`.

```julia
# For Arrays the subset will be of type SubArray
@assert typeof(shuffleobs(rand(4,10))) <: SubArray

# Iterate through all observations in random order
for (x) in eachobs(shuffleobs(X))
    ...
end
```

The optional (keyword) parameter `obsdim` allows one to specify
which dimension denotes the observations. see `LearnBase.ObsDim`
for more detail.

For this function to work, the type of `data` must implement
[`nobs`](@ref) and [`getobs`](@ref). See [`DataSubset`](@ref)
for more information.
"""
shuffleobs(data; obsdim = default_obsdim(data)) =
    shuffleobs(data, convert(LearnBase.ObsDimension,obsdim))

function shuffleobs(data, obsdim)
    datasubset(data, shuffle(1:nobs(data, obsdim)), obsdim)
end
