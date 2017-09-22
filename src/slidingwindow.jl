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

# --------------------------------------------------------------------

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

"""
TODO
"""
struct UnlabeledSlidingWindow{TElem,TData,O} <: SlidingWindow{TElem,TData,O}
    data::TData
    size::Int
    stride::Int
    count::Int
    obsdim::O
end

function slidingwindow(data::T, size::Int, stride::Int, obsdim::O = default_obsdim(data)) where {T,O}
    E = typeof(datasubset(data,1:size,obsdim))
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    UnlabeledSlidingWindow{E,T,O}(data, size, stride, count, obsdim)
end

function slidingwindow(data, size::Int; stride=size, obsdim=default_obsdim(data))
    slidingwindow(data, size, stride, convert(LearnBase.ObsDimension, obsdim))
end

function Base.getindex(A::UnlabeledSlidingWindow, windowindex::Int)
    _, windowrange = _windowsettings(A, windowindex)
    datasubset(A.data, windowrange, A.obsdim)
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

"""
TODO
"""
struct LabeledSlidingWindow{TElem,TData,O,TFun,Exclude} <: SlidingWindow{TElem,TData,O}
    data::TData
    targetfun::TFun
    size::Int
    stride::Int
    count::Int
    offset::Int
    obsdim::O
end

function slidingwindow(f::F, data::T, size::Int, stride::Int, ::Type{Val{Exclude}}=Val{false}, obsdim::O = default_obsdim(data)) where {F,T,Exclude,O}
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    # limit back of sequence until it contains target
    o = 1 + (count-1) * stride
    while maximum(f(o)) > nobs(data)
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

function slidingwindow(f, data, size::Int; stride=size, excludetarget=false, obsdim=default_obsdim(data))
    slidingwindow(f, data, size, stride, Val{excludetarget}, convert(LearnBase.ObsDimension, obsdim))
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
