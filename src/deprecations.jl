@deprecate DataSubset(data::T; obsdim = default_obsdim(data)) where {T} DataSubset(data, 1:nobs(data; obsdim = obsdim))
@deprecate datasubset(data::T; obsdim = default_obsdim(data)) where {T} datasubset(data, 1:nobs(data; obsdim = obsdim))
# discuss if we want to have this
@deprecate getobs(subset::DataSubset) getobs(subset.data, subset.indices)
