abstract type SlidingWindow{TElem,TData,O} <: DataView{TElem,TData} end

Base.parent(A::SlidingWindow) = A.data
Base.length(A::SlidingWindow) = nobs(A)
nobs(A::SlidingWindow) = A.count

# compatibility with nested functions
default_obsdim(A::SlidingWindow) = A.obsdim

function _windowsettings(A::SlidingWindow, windowindex::Int)
    offset = 1 + (windowindex-1) * A.stride
    range = offset:offset+A.size-1
    offset, range
end

function _check_windowargs(data, size::Int, stride::Int, obsdim)
    size > 0 || throw(ArgumentError("Specified window size must be strictly greater than 0. Actual: $size"))
    size <= nobs(data,obsdim) || throw(ArgumentError("Specified window size is too large for the given number of observations"))
    stride > 0 || throw(ArgumentError("Specified stride must be strictly greater than 0. Actual: $stride"))
end

# --------------------------------------------------------------------

struct UnlabeledSlidingWindow{TElem,TData,O} <: SlidingWindow{TElem,TData,O}
    data::TData
    size::Int
    stride::Int
    count::Int
    obsdim::O
end

"""
    slidingwindow(data, size, [stride], [obsdim]) -> UnlabeledSlidingWindow

Return a vector-like view of the `data` for which each element is
a fixed size "window" of `size` adjacent observations. By default
these windows are not overlapping. Note that only complete
windows are included in the output, which implies that it is
possible for excess observations to be omitted from the view.

```julia-repl
julia> A = slidingwindow(1:20, 6)
3-element slidingwindow(::UnitRange{Int64}, 6) with element type SubArray{Int64,1,UnitRange{Int64},Tuple{UnitRange{Int64}},true}:
 [1, 2, 3, 4, 5, 6]
 [7, 8, 9, 10, 11, 12]
 [13, 14, 15, 16, 17, 18]
```

Note that the values of `data` itself are not copied. Instead the
function [`datasubset`](@ref) is called when `getindex` is
invoked. To actually get a copy of the data at some window use
the function [`getobs`](@ref).

```julia-repl
julia> A[1]
6-element SubArray{Int64,1,UnitRange{Int64},Tuple{UnitRange{Int64}},true}:
 1
 ⋮
 6

julia> getobs(A, 1)
6-element Array{Int64,1}:
 1
 ⋮
 6
```

The optional parameter `stride` can be used to specify the
distance between the start elements of each adjacent window.
By default the stride is equal to the window size.

```julia-repl
julia> slidingwindow(1:20, 6, stride = 3)
5-element slidingwindow(::UnitRange{Int64}, 6, stride = 3) with element type SubArray{Int64,1,UnitRange{Int64},Tuple{UnitRange{Int64}},true}:
 [1, 2, 3, 4, 5, 6]
 [4, 5, 6, 7, 8, 9]
 [7, 8, 9, 10, 11, 12]
 [10, 11, 12, 13, 14, 15]
 [13, 14, 15, 16, 17, 18]
```

The optional (keyword) parameter `obsdim` allows one to specify
which dimension denotes the observations. see `LearnBase.ObsDim`
for more detail.
"""
function slidingwindow(data::T, size::Int, stride::Int, obsdim::O = default_obsdim(data)) where {T,O}
    _check_windowargs(data, size, stride, obsdim)
    E = typeof(datasubset(data,1:size,obsdim))
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    UnlabeledSlidingWindow{E,T,O}(data, size, stride, count, obsdim)
end

function slidingwindow(data, size::Int, obsdim::Union{Tuple,ObsDimension})
    # default to using same stride as size
    slidingwindow(data, size, size, obsdim)
end

function slidingwindow(data, size::Int; stride=size, obsdim=default_obsdim(data))
    slidingwindow(data, size, stride, convert(ObsDimension, obsdim))
end

function Base.getindex(A::UnlabeledSlidingWindow, windowindex::Int)
    _, windowrange = _windowsettings(A, windowindex)
    datasubset(A.data, windowrange, A.obsdim)
end

getobs(A::UnlabeledSlidingWindow, indices::AbstractVector) = [getobs(A,i) for i in indices]

# --------------------------------------------------------------------

struct LabeledSlidingWindow{TElem,TData,O,TFun,Exclude} <: SlidingWindow{TElem,TData,O}
    data::TData
    targetfun::TFun
    size::Int
    stride::Int
    count::Int
    offset::Int
    obsdim::O
end

"""
    slidingwindow(f, data, size, [stride], [excludetarget], [obsdim]) -> LabeledSlidingWindow

Return a vector-like view of the `data` for which each element is
a tuple of two elements:

1. A fixed size "window" of `size` adjacent observations. By
   default these windows are not overlapping. This can be
   changed by explicitly specifying a `stride`.

2. A single target (or vector of targets) for the window. The
   content of the target(s) is defined by the label-index
   function `f`.

The label-index function `f` is a unary function that takes the
index of the first observation in the current window and should
return the index (or indices) of the associated target(s) for
that window.

```julia-repl
julia> A = slidingwindow(i->i+6, 1:20, 6)
3-element slidingwindow(::##3#4, ::UnitRange{Int64}, 6) with element type Tuple{...}
 ([1, 2, 3, 4, 5, 6], 7)
 ([7, 8, 9, 10, 11, 12], 13)
 ([13, 14, 15, 16, 17, 18], 19)
```

Note that only complete and in-bound windows are included in the
output, which implies that it is possible for excess observations
to be omitted from the resulting view.

```julia-repl
julia> A = slidingwindow(i->i-1, 1:20, 6)
2-element slidingwindow(::##5#6, ::UnitRange{Int64}, 6) with element type Tuple{...}
 ([7, 8, 9, 10, 11, 12], 6)
 ([13, 14, 15, 16, 17, 18], 12)
```

As hinted above, it is also allowed for `f` to return a vector of
indices. This can be useful for emulating techniques such as
skip-gram.

```julia-repl
julia> data = split("The quick brown fox jumps over the lazy dog")
9-element Array{SubString{String},1}:
 "The"
 "quick"
 ⋮
 "lazy"
 "dog"

julia> A = slidingwindow(i->[i-2:i-1; i+1:i+2], data, 1)
5-element slidingwindow(::##11#12, ::Array{SubString{String},1}, 1) with element type Tuple{...}:
 (["brown"], ["The", "quick", "fox", "jumps"])
 (["fox"], ["quick", "brown", "jumps", "over"])
 (["jumps"], ["brown", "fox", "over", "the"])
 (["over"], ["fox", "jumps", "the", "lazy"])
 (["the"], ["jumps", "over", "lazy", "dog"])
```

Should it so happen that the targets overlap with the features,
then the affected observation(s) will be present in both. To
change this behaviour one can set the optional parameter
`excludetarget = true`. This will remove the target(s) from the
feature window.

```julia-repl
julia> slidingwindow(i->i+2, data, 5, stride = 1, excludetarget = true)
5-element slidingwindow(::##17#18, ::Array{SubString{String},1}, 5, stride = 1) with element type Tuple{...}:
 (["The", "quick", "fox", "jumps"], "brown")
 (["quick", "brown", "jumps", "over"], "fox")
 (["brown", "fox", "over", "the"], "jumps")
 (["fox", "jumps", "the", "lazy"], "over")
 (["jumps", "over", "lazy", "dog"], "the")
```

The optional (keyword) parameter `obsdim` allows one to specify
which dimension denotes the observations. see `LearnBase.ObsDim`
for more detail.
"""
function slidingwindow(f::F, data::T, size::Int, stride::Int, ::Type{Val{Exclude}}=Val{false}, obsdim::O = default_obsdim(data)) where {F,T,Exclude,O}
    _check_windowargs(data, size, stride, obsdim)
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    # limit back of sequence until it contains target
    o = 1 + (count-1) * stride
    while maximum(f(o)) > nobs(data,obsdim)
        count -= 1
        o = 1 + (count-1) * stride
    end
    # limit front of sequence until it contains target
    offset = 0
    o = 1 + offset * stride
    while minimum(f(o)) < 1
        offset += 1
        count -= 1
        o = 1 + offset * stride
    end
    # construct view
    E = typeof((datasubset(data,o:o+size-1,obsdim),datasubset(data,f(o),obsdim)))
    LabeledSlidingWindow{E,T,O,F,Exclude}(data, f, size, stride, count, offset, obsdim)
end

function slidingwindow(f, data, size::Int, stride::Int, obsdim::Union{ObsDimension,Tuple}, ::Type{Val{Exclude}}=Val{false}) where {Exclude}
    slidingwindow(f, data, size, stride, Val{Exclude}, obsdim)
end

function slidingwindow(f::Function, data, size::Int, obsdim::Union{ObsDimension,Tuple}, ::Type{Val{Exclude}}=Val{false}) where {Exclude}
    slidingwindow(f, data, size, size, Val{Exclude}, obsdim)
end

function slidingwindow(f::Function, data, size::Int, ::Type{Val{Exclude}}, obsdim = default_obsdim(data)) where {Exclude}
    slidingwindow(f, data, size, size, Val{Exclude}, obsdim)
end

Base.@pure _toVal(::Type{Val{T}}) where {T} = Val{T}
Base.@pure _toVal(T) = Val{T}
function slidingwindow(f, data, size::Int; stride=size, excludetarget=Val{false}, obsdim=default_obsdim(data))
    slidingwindow(f, data, size, stride, _toVal(excludetarget), convert(ObsDimension, obsdim))
end

function Base.getindex(A::LabeledSlidingWindow{TElem,TData,O,TFun,Exclude}, wi::Int) where {TElem,TData,O,TFun,Exclude}
    windowindex = wi + A.offset
    offset, windowrange = _windowsettings(A, windowindex)
    windowindices = if Exclude
        setdiff(windowrange, A.targetfun(offset))
    else
        windowrange
    end
    X = datasubset(A.data, windowindices, A.obsdim)
    Y = datasubset(A.data, A.targetfun(offset), A.obsdim)
    X, Y
end

function getobs(A::LabeledSlidingWindow, indices::AbstractVector)
    map(i->getobs(A,i), indices)
end

function getobs(A::LabeledSlidingWindow{<:NTuple{<:Any,Tuple}}, i::Int)
    map(a->getobs.(a), A[i])
end

# --------------------------------------------------------------------

function ShowItLikeYouBuildIt.showarg(io::IO, A::SlidingWindow)
    print(io, "slidingwindow(")
    if A isa LabeledSlidingWindow
        showarg(io, A.targetfun)
        print(io, ", ")
    end
    showarg(io, parent(A))
    print(io, ", ")
    print(io, A.size)
    if A.stride != A.size
        print(io, ", ")
        print(io, "stride = ", A.stride)
    end
    if A.obsdim != default_obsdim(A.data)
        print(io, ", ")
        print(io, "obsdim = ", obsdim_string(A.obsdim))
    end
    print(io, ')')
end

Base.summary(A::SlidingWindow) = summary_build(A)

# --------------------------------------------------------------------
# batches of sliding windows

const WindowBatchView{TElem,O} = BatchView{TElem,<:SlidingWindow,O}

function BatchView(data::T, size::Int, count::Int, obsdim::O = default_obsdim(data), upto::Bool = false) where {T<:SlidingWindow,O}
    nsize, ncount = _compute_batch_settings(data, size, count, obsdim, upto)
    E = typeof(_getwindowbatch(data, nsize, 1))
    BatchView{E,T,O}(data, nsize, ncount, obsdim)
end

function Base.getindex(A::WindowBatchView, batchindex::Int)
    _getwindowbatch(parent(A), A.size, batchindex)
end

function Base.getindex(A::WindowBatchView, batchindices::AbstractVector)
    [A[bi] for bi in batchindices]
end

# --------------------------------------------------------------------

getobs(A::BatchView{T,<:UnlabeledSlidingWindow}) where {T} = map(x->getobs.(x), A)
getobs(A::BatchView{T,<:UnlabeledSlidingWindow}, i::AbstractVector) where {T} = map(x->getobs.(x), A)
getobs(A::BatchView{T,<:UnlabeledSlidingWindow}, i::Int) where {T} = getobs.(A[i])

function _getwindowbatch(W::UnlabeledSlidingWindow, batchsize::Int, batchindex::Int)
    batchrange = _batchrange(batchsize, batchindex)
    windowranges = [_windowsettings(W, windowindex)[2] for windowindex in batchrange]
    [datasubset(W.data, [windowranges[wi][oi] for wi in 1:batchsize], W.obsdim) for oi in 1:W.size]
end

# --------------------------------------------------------------------

function _getwindowbatch(A::LabeledSlidingWindow{TElem,TData,O,TFun,Exclude}, batchsize::Int, batchindex::Int) where {TElem,TData,O,TFun,Exclude}
    batchrange = _batchrange(batchsize, batchindex)
    fltr = if Exclude
        s -> (A.targetfun(s[1]), setdiff(s[2], A.targetfun(s[1])))
    else
        s -> (A.targetfun(s[1]), s[2])
    end
    windowsettings = [fltr(_windowsettings(A, windowindex)) for windowindex in batchrange]
    ([datasubset(A.data, [windowsettings[wi][2][oi] for wi in 1:batchsize], A.obsdim)
        for oi in 1:length(windowsettings[1][2])],
     [datasubset(A.data, [windowsettings[wi][1][oi] for wi in 1:batchsize], A.obsdim)
        for oi in 1:length(windowsettings[1][1])])
end
