"""
    oversample([f], data, [fraction = 1], [shuffle = true], [obsdim])

Generate a re-balanced version of `data` by repeatedly sampling
existing observations in such a way that every class will have at
least `fraction` times the number observations of the largest
class. This way, all classes will have a minimum number of
observations in the resulting data set relative to what largest
class has in the given (original) `data`.

As an example, by default (i.e. with `fraction = 1`) the
resulting dataset will be near perfectly balanced. On the other
hand, with `fraction = 0.5` every class in the resulting data
with have at least 50% as many observations as the largest class.

The convenience parameter `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the repeated samples will be together at the
end, sorted by class. Defaults to `true`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?ObsDim` for more information.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# oversample the class "a" to match "b"
X_bal, Y_bal = oversample((X,Y))

# this results in a bigger dataset with repeated data
@assert size(X_bal) == (3,8)
@assert length(Y_bal) == 8

# now both "a", and "b" have 4 observations each
@assert sum(Y_bal .== "a") == 4
@assert sum(Y_bal .== "b") == 4
```

For this function to work, the type of `data` must implement
[`nobs`](@ref) and [`getobs`](@ref). For example, the following
code allows `oversample` to work on a `DataTable`.

```julia
# Make DataTables.jl work
LearnBase.getobs(data::DataTable, i) = data[i,:]
StatsBase.nobs(data::DataTable) = nrow(data)
```

You can use the parameter `f` to specify how to extract or
retrieve the targets from each observation of the given `data`.
Note that if `data` is a tuple, then it will be assumed that the
last element of the tuple contains the targets and `f` will be
applied to each observation in that element.

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

julia> getobs(oversample(row->row[:Y], data))
8×3 DataTables.DataTable
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.0997825 │ 0.000341996 │ a │
│ 2   │ 0.505208  │ 0.11202     │ b │
│ 3   │ 0.226582  │ 0.0443222   │ a │
│ 4   │ 0.0997825 │ 0.000341996 │ a │
│ 5   │ 0.504629  │ 0.722906    │ b │
│ 6   │ 0.522172  │ 0.245457    │ b │
│ 7   │ 0.226582  │ 0.0443222   │ a │
│ 8   │ 0.933372  │ 0.812814    │ b │
```

see [`DataSubset`](@ref) for more information on data subsets.

see also [`undersample`](@ref) and [`stratifiedobs`](@ref).
"""
oversample(data; fraction=1, shuffle=true, obsdim=default_obsdim(data)) =
    oversample(identity, data, fraction, shuffle, obsdim)

oversample(data, shuffle::Bool, obsdim=default_obsdim(data)) =
    oversample(identity, data, shuffle, obsdim)

oversample(data, fraction::Real, obsdim=default_obsdim(data)) =
    oversample(identity, data, fraction, true, obsdim)

oversample(data, fraction::Real, shuffle::Bool, obsdim=default_obsdim(data)) =
    oversample(identity, data, fraction, shuffle, obsdim)

# in order to disambiguate methods
oversample(f::Function, data; fraction=1, shuffle=true, obsdim=default_obsdim(data)) =
    oversample(f, data, fraction, shuffle, obsdim)

oversample(f::Function, data, shuffle::Bool, obsdim=default_obsdim(data)) =
    oversample(f, data, 1, shuffle, obsdim)

function oversample(f::Function, data, fraction::Real, shuffle::Bool=true, obsdim=default_obsdim(data))
    allowcontainer(oversample, data) || throw(MethodError(oversample, (f,data,shuffle,obsdim)))
    lm = labelmap(eachtarget(f, data; obsdim = obsdim))

    maxcount = maximum(length, values(lm))
    fraccount = round(Int, fraction * maxcount)

    # firstly we will start by keeping everything
    inds = collect(1:nobs(data; obsdim = obsdim))
    sizehint!(inds, nlabel(lm)*maxcount)

    for (lbl, inds_for_lbl) in lm
        num_extra_needed = fraccount - length(inds_for_lbl)
        while num_extra_needed > length(inds_for_lbl)
            num_extra_needed -= length(inds_for_lbl)
            append!(inds, inds_for_lbl)
        end
        if num_extra_needed > 0
            append!(inds, sample(inds_for_lbl, num_extra_needed; replace=false))
        end
    end

    shuffle && shuffle!(inds)
    datasubset(data, inds)
end

"""
    undersample([f], data, [shuffle = false], [obsdim])

Generate a class-balanced version of `data` by subsampling its
observations in such a way that the resulting number of
observations will be the same number for every class. This way,
all classes will have as many observations in the resulting data
set as the smallest class has in the given (original) `data`.

The convenience parameter `shuffle` determines if the
resulting data will be shuffled after its creation; if it is not
shuffled then all the observations will be in their original
order. Defaults to `false`.

The optional parameter `obsdim` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of `data`. See `?ObsDim` for more information.

```julia
# 6 observations with 3 features each
X = rand(3, 6)
# 2 classes, severely imbalanced
Y = ["a", "b", "b", "b", "b", "a"]

# subsample the class "b" to match "a"
X_bal, Y_bal = undersample((X,Y))

# this results in a smaller dataset
@assert size(X_bal) == (3,4)
@assert length(Y_bal) == 4

# now both "a", and "b" have 2 observations each
@assert sum(Y_bal .== "a") == 2
@assert sum(Y_bal .== "b") == 2
```

For this function to work, the type of `data` must implement
[`nobs`](@ref) and [`getobs`](@ref). For example, the following
code allows `undersample` to work on a `DataTable`.

```julia
# Make DataTables.jl work
LearnBase.getobs(data::DataTable, i) = data[i,:]
StatsBase.nobs(data::DataTable) = nrow(data)
```

You can use the parameter `f` to specify how to extract or
retrieve the targets from each observation of the given `data`.
Note that if `data` is a tuple, then it will be assumed that the
last element of the tuple contains the targets and `f` will be
applied to each observation in that element.

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

julia> getobs(undersample(row->row[:Y], data))
4×3 DataTables.DataTable
│ Row │ X1        │ X2          │ Y │
├─────┼───────────┼─────────────┼───┤
│ 1   │ 0.226582  │ 0.0443222   │ a │
│ 2   │ 0.504629  │ 0.722906    │ b │
│ 3   │ 0.522172  │ 0.245457    │ b │
│ 4   │ 0.0997825 │ 0.000341996 │ a │
```

see [`DataSubset`](@ref) for more information on data subsets.

see also [`oversample`](@ref) and [`stratifiedobs`](@ref).
"""
undersample(data; shuffle=false, obsdim=default_obsdim(data)) =
    undersample(identity, data, shuffle, obsdim)

undersample(data, shuffle::Bool, obsdim=default_obsdim(data)) =
    undersample(identity, data, shuffle, obsdim)

undersample(f, data; shuffle=false, obsdim=default_obsdim(data)) =
    undersample(f, data, shuffle, obsdim)

function undersample(f, data, shuffle::Bool, obsdim=default_obsdim(data))
    allowcontainer(undersample, data) || throw(MethodError(undersample, (f,data,shuffle,obsdim)))
    lm = labelmap(eachtarget(f, data, obsdim))
    mincount = minimum(length, values(lm))

    inds = Int[]
    sizehint!(inds, nlabel(lm)*mincount)

    for (lbl, inds_for_lbl) in lm
        append!(inds, sample(inds_for_lbl, mincount; replace=false))
    end

    shuffle ? shuffle!(inds) : sort!(inds)
    datasubset(data, inds)
end

# Make sure the R people find the functionality
@deprecate upsample(args...; kw...) oversample(args...; kw...)
@deprecate downsample(args...; kw...) undersample(args...; kw...)
