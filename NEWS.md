# v0.2.0

- drop julia 0.5 support.

Small changes:

- rework how `FoldsView` is displayed.

- make `BatchView` only print the "unused datapoints" warning once.

- add test dependency on `ReferenceTests.jl`.

# v0.1.2

Small changes:

- reworked how data views and data iterators are displayed

- added `maxsize` keyword argument to `eachbatch` and `batchview`

# v0.1

New features:

- added `stratifiedobs` to perform partitioning using stratified
  sampling without replacement.
