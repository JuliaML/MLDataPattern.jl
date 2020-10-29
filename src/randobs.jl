# TODO: allow passing a rng as first parameter?

"""
    randobs(data, [n], [obsdim])

Pick a random observation or a batch of `n` random observations
from `data`.

The optional (keyword) parameter `obsdim` allows one to specify
which dimension denotes the observations. see `LearnBase.ObsDim`
for more detail.

For this function to work, the type of `data` must implement
[`nobs`](@ref) and [`getobs`](@ref).
"""
randobs(data; obsdim = default_obsdim(data)) =
    randobs(data, 1, obsdim)

randobs(data, n; obsdim = default_obsdim(data)) =
    randobs(data, n, obsdim)

randobs(data, n, obsdim) =
    getobs(data, rand(1:nobs(data, obsdim), n), obsdim)
