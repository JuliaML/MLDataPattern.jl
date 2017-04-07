__precompile__()
module MLDataPattern

using LearnBase
using MLLabelUtils
using Compat

using LearnBase: ObsDimension
import LearnBase: nobs, getobs, getobs!, gettarget, gettargets, targets, datasubset, default_obsdim

export

    ObsDim,

    nobs,
    getobs,
    getobs!,

    randobs,

    DataSubset,
    datasubset,

    shuffleobs,
    splitobs,

    ObsView,
    BatchView,
    obsview,
    batchview,
    batchsize,

    FoldsView,
    kfolds,
    leaveout

include("container.jl")
include("datasubset.jl")
include("randobs.jl")
include("shuffleobs.jl")
include("splitobs.jl")
include("dataview.jl")
include("folds.jl")

end # module
