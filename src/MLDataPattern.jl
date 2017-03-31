__precompile__()
module MLDataPattern

using LearnBase
using MLLabelUtils
using Compat

using LearnBase: ObsDimension, obs_dim
import LearnBase: nobs, getobs, getobs!, gettarget, gettargets, targets, datasubset, default_obsdim

export

    nobs,
    getobs,
    getobs!,

    DataSubset,
    datasubset,

    randobs

include("container.jl")
include("datasubset.jl")
include("randobs.jl")

end # module
