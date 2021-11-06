@deprecate DataSubset(data::T; obsdim = default_obsdim(data)) where {T} DataSubset(data::T, 1:nobs(data; obsdim = obsdim))
@deprecate datasubset(data::T; obsdim = default_obsdim(data)) where {T} datasubset(data::T, 1:nobs(data; obsdim = obsdim))
# discuss if we want to have this
@deprecate getobs(subset::DataSubset) getobs(subset, 1:nobs(subset))