__precompile__()
module MLDataPattern

using LearnBase
using MLLabelUtils
using Compat

using LearnBase: ObsDimension, obs_dim
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

    FoldsView,
    kfolds,
    leaveout

include("container.jl")
include("datasubset.jl")
include("randobs.jl")
include("shuffleobs.jl")
include("splitobs.jl")
include("folds.jl")

end # module
