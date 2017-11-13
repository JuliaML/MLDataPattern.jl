# v0.3.0

New features:

- added `BalancedObs` iterator to allow for balanced sampling
  from labeled data container that with skewed label
  distributions.

# v0.2.0

- drop julia 0.5 support.

New features:

- added `slidingwindow` to help prepare sequence data for
  training.

Small changes:

- rework how `FoldsView` is displayed.

- make `BatchView` only print the "unused datapoints" warning once.

- add test dependency on `ReferenceTests.jl`.

Fixes:

- disallow certain functions on `DataView` that don't work the
  way a user would expect.

# v0.1.2

Small changes:

- reworked how data views and data iterators are displayed

- added `maxsize` keyword argument to `eachbatch` and `batchview`

# v0.1

New features:

- added `stratifiedobs` to perform partitioning using stratified
  sampling without replacement.
