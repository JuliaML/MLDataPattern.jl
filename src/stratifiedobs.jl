splitobs{T,I<:Integer}(lm::Dict{T,Vector{I}}; at = 0.7) =
    splitobs(lm, at)

function splitobs{T,I<:Integer}(lm::Dict{T,Vector{I}}, at::AbstractFloat)
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
        i1, i2 = splitobs(indices, at)
        append!(idx1, i1)
        append!(idx2, i2)
    end
    idx1, idx2
end

# we use @generated because we compute "N+1"
@generated function splitobs{T,I<:Integer,N}(lm::Dict{T,Vector{I}}, at::NTuple{N,AbstractFloat})
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
        @nexprs $(N)   i -> sizehint!(idx_i, ceil(Int, n*at[i] + k))
        sizehint!($(Symbol(:idx_, Symbol(N+1))), ceil(Int, n*(1-sum(at)) + k))
        # loop through all label indices
        for indices in values(lm)
            tup = splitobs(indices, at)
            @nexprs $(N+1) i -> append!(idx_i, tup[i])
        end
        # return a tuple of all indices vectors
        @ntuple $(N+1) idx
    end
end

function stratifiedobs(data; p = 0.7, shuffle = true, obsdim = default_obsdim(data))
    stratifiedobs(identity, data, p, shuffle, convert(ObsDimension, obsdim))
end

function stratifiedobs(f, data; p = 0.7, shuffle = true, obsdim = default_obsdim(data))
    stratifiedobs(f, data, p, shuffle, convert(ObsDimension, obsdim))
end

function stratifiedobs(data, p::AbstractFloat, args...)
    stratifiedobs(identity, data, p, args...)
end

function stratifiedobs{N}(data, p::NTuple{N,AbstractFloat}, args...)
    stratifiedobs(identity, data, p, args...)
end

function stratifiedobs(f, data, p::Union{NTuple,AbstractFloat}, shuffle::Bool = true, obsdim = default_obsdim(data))
    # The given data is always shuffled to qualify as performing
    # stratified sampling without replacement.
    data_shuf = shuffleobs(data, obsdim)
    idx_tup = splitobs(labelmap(eachtarget(f, data_shuf, obsdim)), p)
    # Setting the parameter "shuffle = false" specifies that the
    # classes are ordered in the resulting subsets respectively.
    shuffle && foreach(shuffle!, idx_tup)
    map(idx -> datasubset(data_shuf, idx, obsdim), idx_tup)
end
