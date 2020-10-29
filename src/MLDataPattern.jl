module MLDataPattern
using StatsBase
using LearnBase
using MLLabelUtils

import LearnBase: getobs, getobs!, gettarget, gettargets, targets, datasubset, default_obsdim

using Base.Cartesian
using Random
using SparseArrays

export

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

    SlidingWindow,
    slidingwindow,

    # targets,
    # eachtarget,

    stratifiedobs,

    oversample,
    undersample,
    upsample,
    downsample,

    FoldsView,
    kfolds,
    leaveout,

    RandomObs,
    BalancedObs,
    RandomBatches,
    BufferGetObs,
    eachobs,
    eachbatch

# obsdim_string(::ObsDim.First) = ":first"
# obsdim_string(::ObsDim.Last) = ":last"
# obsdim_string(::ObsDim.Constant{D}) where {D} = string(D)
# obsdim_string(::ObsDim.Undefined) = "\"NA\""
# obsdim_string(obsdim::Tuple) = string("(", join(map(obsdim_string, obsdim), ", "), ")")

include("container.jl")
include("datasubset.jl")
include("randobs.jl")
include("shuffleobs.jl")
include("splitobs.jl")
include("dataview.jl")
include("slidingwindow.jl")
include("targets.jl")
include("stratifiedobs.jl")
include("resample.jl")
include("folds.jl")
include("dataiterator.jl")

end # module
