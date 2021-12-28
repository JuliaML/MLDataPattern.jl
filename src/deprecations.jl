@deprecate DataSubset(data; obsdim = default_obsdim(data)) DataSubset(data, 1:nobs(data; obsdim = obsdim))
