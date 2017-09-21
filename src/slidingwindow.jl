struct SlidingWindow{TElem,TData,O,TFun} <: LearnBase.DataView{TElem,TData}
    data::TData
    targetfun::TFun
    size::Int
    stride::Int
    count::Int
    obsdim::O
end

const UnlabeledSlidingWindow{TElem,TData,O} = SlidingWindow{TElem,TData,O,Void}

function slidingwindow(data::T, size::Int, stride::Int, obsdim::O = default_obsdim(data)) where {T,O}
    E = typeof(datasubset(data,1:1+size,obsdim))
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    SlidingWindow{E,T,O,Void}(data, nothing, size, stride, count, obsdim)
end

function slidingwindow(f::F, data::T, size::Int, stride::Int, obsdim::O = default_obsdim(data)) where {F,T,O}
    E = typeof((datasubset(data,1:1+size,obsdim),datasubset(data,f(1),obsdim)))
    count = floor(Int, (nobs(data,obsdim) - size + stride) / stride)
    offset = 1 + (count-1) * stride
    while maximum(f(offset)) > nobs(data)
        count -= 1
        offset = 1 + (count-1) * stride
    end
    SlidingWindow{E,T,O,F}(data, f, size, stride, count, obsdim)
end

slidingwindow(data, size::Int; stride=size, obsdim=default_obsdim(data)) =
    slidingwindow(data, size, stride, convert(LearnBase.ObsDimension, obsdim))

slidingwindow(f, data, size::Int; stride=size, obsdim=default_obsdim(data)) =
    slidingwindow(f, data, size, stride, convert(LearnBase.ObsDimension, obsdim))

Base.parent(A::SlidingWindow) = A.data
Base.length(A::SlidingWindow) = nobs(A)
nobs(A::SlidingWindow) = A.count

function _windowsettings(A::SlidingWindow, windowindex::Int)
    offset = 1 + (windowindex-1) * A.stride
    range = offset:offset+A.size-1
    offset, range
end

function Base.getindex(A::UnlabeledSlidingWindow, windowindex::Int)
    _, windowrange = _windowsettings(A, windowindex)
    datasubset(A.data, windowrange, A.obsdim)
end

function Base.getindex(A::SlidingWindow, windowindex::Int)
    offset, windowrange = _windowsettings(A, windowindex)
    X = datasubset(A.data, windowrange, A.obsdim)
    Y = datasubset(A.data, A.targetfun(offset), A.obsdim)
    X, Y
end

# compatibility with nested functions
default_obsdim(A::SlidingWindow) = A.obsdim

function ShowItLikeYouBuildIt.showarg(io::IO, A::SlidingWindow)
    print(io, "slidingwindow(")
    if A.targetfun != nothing
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

const WindowBatchView{TElem,O} = BatchView{TElem,<:SlidingWindow,O}
const UnlabeledWindowBatchView{TElem,O} = BatchView{TElem,<:UnlabeledSlidingWindow,O}

function BatchView(data::T, size::Int, count::Int, obsdim::O = default_obsdim(data), upto::Bool = false) where {T<:SlidingWindow,O}
    nsize, ncount = _compute_batch_settings(data, size, count, obsdim, upto)
    E = Any
    BatchView{E,T,O}(data, nsize, ncount, obsdim)
end

function Base.getindex(A::WindowBatchView, batchindex::Int)
    W = parent(A)
    batchrange = _batchrange(A.size, batchindex)
    windowranges = [_windowsettings(W, windowindex)[2] for windowindex in batchrange]
    [datasubset(W.data, [windowranges[wi][oi] for wi in 1:A.size], W.obsdim) for oi in 1:W.size]
end

function Base.getindex(A::WindowBatchView, batchindices::AbstractVector)
    [A[bi] for bi in batchindices]
end
