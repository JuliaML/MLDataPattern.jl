.. _subsets:

Data Subsets
===========================

It is a common requirement in Machine Learning related
experiments to partition some data set in one way or the other.
At its essence, data partitioning can be thought of as a process
that assigns observations to one or more subsets of the original
data. This abstraction is also true for other important and
widely used data access pattern in machine learning (e.g. over-
and under-sampling of a labeled data set).

In other words, the core problem that needs to be addressed
efficiently, is how to create and represent a **data subset** in
a generic way. Once we can subset arbitrary index-based data,
more complicated tasks such as data *partitioning*, *shuffling*,
or *resampling* can be expressed through data subsetting in a
coherent manner.

Before we move on, let us quickly clarify what exactly we mean
when we talk about a data "subset". We don't think about the term
"subset" in the mathematical sense of the word. Instead, when we
attempt to subset some data source, what we are really interested
in, is a representation (aka. subset) of a specific sequence of
observations from the original data source. We specify which
observations we want to be part of this subset, by using
observation-indices from the set :math:`I = \{1,2,...,N\}`. Here
`N` is the total number of observations in our data source. This
interpretation of "subset" implies the following:

1. We can only subset data sources that are considered data
   container. Furthermore, a subset of a data container is again
   considered a data container.

2. When specifying a subset, the order of the requested
   observation-indices matter. That means that different index
   permutations will cause conceptually different "subsets".

3. A subset can contain the same exact observation for an
   arbitrary number of times (including zero). Furthermore, an
   observation can be part of multiple distinct subsets.

We will spend the rest of this document discussing how to use
this package to create data subsets and how to interact with
them. After introducing the basics, we will go over the multiple
high-level functions that create data subsets for you. These
include splitting your data into train and test portions,
shuffling your data, and resampling your data using a k-folds
partitioning scheme.

Subsetting a Data Container
---------------------------

We have seen before that when confronted with a **data
container**, nesting various subsetting operations really just
breaks down to keeping track of the observation-*indices*. This
in turn is much cheaper than copying observation-*values* around
needlessly (see :ref:`background` for an in-depth discussion).

Ideally, when we "subset" a data container, what we want is a
lazy representation of that subset. In other words, we would like
to avoid copying the values of our data set around until we
actually need it. To that end, we provide the function
:func:`datasubset`, which tries to choose the most appropriate
type of subset for the given data container.

.. function:: datasubset(data, [idx], [obsdim])

   Returns a lazy subset of the observations in `data` that
   correspond to the given index/indices in `idx`. No data will
   be copied except of the indices

   This function is similar to calling the constructor for
   :class:`DataSubset`, with the main difference that
   :func:`datasubset` will return a ``SubArray`` if the type of
   `data` is an ``Array`` or ``SubArray``. Furthermore, this
   function can be extended for custom types of `data` that also
   want to provide their own subset-type.

   The returned subset will in general not be of the same type as
   the underlying observations it represents. If you want to
   query the actual observations corresponding to the given
   indices in their true form, use :func:`getobs` instead.

   :param data: The object representing a data container.

   :param idx: \
        Optional. The index or indices of the observation(s) in
        `data` that should be part of the subset. Can be of type
        ``Int`` or some subtype ``AbstractVector{Int}``. Defaults
        to ``1:nobs(data,obsdim)``

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: An object representing a lazy subset of `data` for
            the observation-indices in `idx`. The type of the
            return value depends on the type of `data`.

Out of the box, this package provides *custom* support for all
subtypes of ``AbstractArray``. With the exception of sparse
arrays, we represent all subsets of arrays in the form of a
``SubArray``. To give a concrete example of what we mean, let us
consider the following random matrix ``X``. We will think about
it as a small data set that has 4 observations with 2 features
each.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> datasubset(X, 2) # single observation at index 2
   2-element SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}:
    0.933372
    0.522172

   julia> datasubset(X, [2,4]) # batch of 2 observations
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.0443222
    0.522172  0.722906

If there is more than one array dimension, all but the
observation dimension are implicitly assumed to be features (i.e.
part of that observation). As you can see in the example above,
the default assumption is that the last array dimension
enumerates the observations. This can be overwritten by
explicitly specifying the ``obsdim``. In the following code
snippet we treat ``X`` as a data set that has 2 observations with
4 features each.

.. code-block:: jlcon

   julia> datasubset(X, 2, ObsDim.First())
   4-element SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    0.504629
    0.522172
    0.0997825
    0.722906

   julia> datasubset(X, 2, obsdim = 1)
   4-element SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    0.504629
    0.522172
    0.0997825
    0.722906

Note how ``obsdim`` can either be provided using a type-stable
positional argument from the namespace ``ObsDim``, or by using a
more flexible and convenient keyword argument. For more take a
look at :ref:`obsdim`.

Remember that every data subset - which includes ``SubArray`` -
is again a fully qualified data container. As such, it supports
both :func:`nobs` and :func:`getobs`.

.. code-block:: jlcon

   julia> mysubset = datasubset(X, [2,4]) # batch of 2 observations
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.0443222
    0.522172  0.722906

   julia> nobs(mysubset)
   2

   julia> getobs(mysubset)
   2×2 Array{Float64,2}:
    0.933372  0.0443222
    0.522172  0.722906

Because a ``SubArray`` is also a data container, it can be
subsetted even further by using :func:`datasubset` again. The
result of which will be a new ``SubArray`` into the original data
container ``X``. As such it will use the accumulated indices of
both subsetting steps. In other words, while subsetting
operations can be nested, they will be combined into a single
layer (i.e. you don't want a subset of a subset of a subset
represented as nested types)

.. code-block:: jlcon

   julia> datasubset(mysubset, 1) # will still be a view into X
   2-element SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}:
    0.933372
    0.522172

It is also possible to link multiple different data containers
together on an per-observation level. This way they can be
subsetted as one coherent unit. To do that, simply put all the
relevant data container into a single ``Tuple``, before passing
it to :func:`datasubset` (or any other function that expect a
data container). The return value will then be a ``Tuple`` of the
same length, with the resulting data subsets in the same
tuple position.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> y = rand(4)
   4-element Array{Float64,1}:
    0.812814
    0.245457
    0.11202
    0.000341996

   julia> datasubset((X,y), 2) # single observation at index 2
   ([0.933372,0.522172],0.24545709827626805)

   julia> Xs, ys = datasubset((X,y), [2,4]) # batch of 2 observations
   ([0.933372 0.0443222; 0.522172 0.722906], [0.245457,0.000341996])

   julia> Xs
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.0443222
    0.522172  0.722906

   julia> ys
   2-element SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}:
    0.245457
    0.000341996

It is worth pointing out, that the tuple elements (i.e. data
container) need not be of the same type, nor of the same shape.
You can observe this in the code above, where ``X`` is a
``Matrix`` while ``y`` is a ``Vector``. Note, however, that all
tuple elements must be data containers themselves. Furthermore,
they all must contain the same exact number of observations. This
is required, even if the requested observation-index would be
in-bounds for each data container individually.

.. code-block:: jlcon

   julia> datasubset((rand(3), rand(4)), 2)
   ERROR: DimensionMismatch("all data container must have the same number of observations")
   [...]

When grouping data containers in a ``Tuple``, it is of course
possible to specify the ``obsdim`` for each data container. If
all data container share the same observation dimension, it
suffices to specify it once.

.. code-block:: jlcon

   julia> Xs, ys = datasubset((X,y), [2,4], obsdim = :last);

   julia> Xs
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.0443222
    0.522172  0.722906

   julia> ys
   2-element SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}:
    0.245457
    0.000341996

Note that if ``obsdim`` is specified as a ``Tuple``, then it
needs to have the same number of elements as the ``Tuple`` of
data containers.

.. code-block:: jlcon

   julia> Xs, ys = datasubset((X,y), [2,4], obsdim = (2,1));

   julia> Xs
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.0443222
    0.522172  0.722906

   julia> ys
   2-element SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}:
    0.245457
    0.000341996

Multiple ``obsdim`` can of course also be specified using
type-stable positional arguments.

.. code-block:: jlcon

   julia> Xs, ys = datasubset((X',y), [2,4], (ObsDim.First(),ObsDim.Last())); # note the transpose

   julia> Xs
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}:
    0.933372   0.522172
    0.0443222  0.722906

   julia> ys
   2-element SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}:
    0.245457
    0.000341996

The DataSubset Type
------------------------------

So far we have only considered subsetting data container of type
``Array``. However, what if we want to subset some other data
container that does not implement the ``AbstractArray``
interface? Naturally, we can't just use ``SubArray`` to represent
those subsets. For that reason we provide a generic type
:class:`DataSubset`, that serves as the default subset type for
every data container that does not implement their own methods
for :func:`datasubset`.

.. class:: DataSubset

   Used as the default type to represent a subset of some
   arbitrary data container. Its main task is to keep track of
   which observation-indices the subset spans. As such it is
   designed in a way that makes sure that subsequent subsettings
   are accumulated without needing to access the actual data.

   The main purpose for the existence of :class:`DataSubset` is
   to delay data-access and -movement until an actual batch of
   data (or single observation) is needed for some computation.
   This is particularily useful when the data is not located in
   memory, but on the hard drive or some remote location. In such
   a scenario one wants to load the required data only when
   needed.

.. function:: DataSubset(data, [idx], [obsdim]) -> DataSubset

   Create an instance of :class:`DataSubset` that will represent
   a lazy subset of the observations in `data` corresponding to
   the given index/indices in `idx`. No data will be copied
   except of the indices.

   If `data` is a :class:`DataSubset`, then the indices of the
   subset will be combined with `idx` and consequently an
   accumulated :class:`DataSubset` will be created and returned.

   In general we advice to use :func:`datasubset` instead of
   calling :func:`DataSubset` directly. This is because
   :func:`datasubset` will only invoke :func:`DataSubset` if
   there is no alternative choice of subset-type known for the
   given `data`.

   :param data: The object representing a data container.

   :param idx: \
        Optional. The index or indices of the observation(s) in
        `data` that should be part of the subset. Can be of type
        ``Int`` or some subtype ``AbstractVector{Int}``. Defaults
        to ``1:nobs(data,obsdim)``

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

The type :class:`DataSubset` can be used to represent a subset of
any type of data container. This even includes arrays, which we
have seen provide their own special type of subset.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> DataSubset(X, 2) # single observation at index 2
   MLDataPattern.DataSubset{Array{Float64,2},Int64,LearnBase.ObsDim.Last}
    1 observations

   julia> DataSubset(X, [2, 4]) # batch of 2 observations
   MLDataPattern.DataSubset{Array{Float64,2},Array{Int64,1},LearnBase.ObsDim.Last}
    2 observations

As you can see, a :class:`DataSubset` does not tell you a lot of
information about the observations it represents. The reason for
this is that it was designed around the requirement of not
needlessly accessing actual data unless requested using
:func:`getobs`. That said, remember that every data subset is
also a fully qualified data container. As such, it supports both
:func:`nobs` and :func:`getobs`.

.. code-block:: jlcon

   julia> mysubset = DataSubset(X, [2, 4]) # batch of 2 observations
   MLDataPattern.DataSubset{Array{Float64,2},Array{Int64,1},LearnBase.ObsDim.Last}
    2 observations

   julia> nobs(mysubset)
   2

   julia> getobs(mysubset) # request the data it represents
   2×2 Array{Float64,2}:
    0.933372  0.0443222
    0.522172  0.722906

The real strength of the :class:`DataSubset` type (or any data
subset really), is that it can be subsetted even further. The
result of which will be a new :class:`DataSubset` into the
original data container ``X`` that uses the accumulated indices.
In other words, while subsetting operations can be nested, they
will be combined into a single layer (i.e. you don’t want a
subset of a subset of a subset represented as nested types)

.. code-block:: jlcon

   julia> mysubset2 = DataSubset(mysubset, 2) # second observation of mysubset
   MLDataPattern.DataSubset{Array{Float64,2},Int64,LearnBase.ObsDim.Last}
    1 observations

   julia> getobs(mysubset2) # request the data it represents
   2-element Array{Float64,1}:
    0.0443222
    0.722906

As you can see in the example above, :class:`DataSubset` also
stores the utilized ``obsdim``. Because we are using an ``Array``
as example data container, the default assumption is that the
last array dimension enumerates the observations. This can be
overwritten by explicitly specifying the ``obsdim``. As always,
the ``obsdim`` can be specified in a type-stable manner using a
positional argument, or by using a more convenient keyword
argument.

.. code-block:: jlcon

   julia> mysubset = DataSubset(X', 2, obsdim = 1) # note the transpose
   MLDataPattern.DataSubset{Array{Float64,2},Int64,LearnBase.ObsDim.Constant{1}}
    1 observations

   julia> getobs(mysubset)
   2-element Array{Float64,1}:
     0.933372
     0.522172

It is worth pointing out that :class:`DataSubset` remembers the
specified ``obsdim``, which means that it is not required to
specify it again for subsequent data access pattern. In contrast
to this, a ``SubArray`` does not have the means to remember it,
and as such one needs to specify the ``obsdim`` every time.

It is also possible to link multiple different data containers
together on an per-observation level. This way they can be
subsetted as one coherent unit. To do that, simply put all the
relevant data container into a single ``Tuple``, before passing
it to :func:`DataSubset`.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> y = rand(4)
   4-element Array{Float64,1}:
    0.812814
    0.245457
    0.11202
    0.000341996

   julia> Xs, ys = DataSubset((X,y), [2,4]); # batch of 2 observations
   (MLDataPattern.DataSubset{Array{Float64,2},Array{Int64,1},LearnBase.ObsDim.Last}
     2 observations,
    MLDataPattern.DataSubset{Array{Float64,1},Array{Int64,1},LearnBase.ObsDim.Last}
     2 observations)

   julia> getobs(Xs)
   2×2 Array{Float64,2}:
    0.933372  0.0443222
    0.522172  0.722906

   julia> getobs(ys)
   2-element Array{Float64,1}:
    0.245457
    0.000341996

Note something subtle but important in the code snippet above.
The constructor :func:`DataSubset` does not return a
:class:`DataSubset` when it is called with a tuple of data
containers. Instead, it maps the constructor onto each data
container individually. Thus if we invoke :func:`DataSubset` with
a ``Tuple``, it will return a ``Tuple`` of :class:`DataSubset`.

Support for Custom Types
----------------------------------

We have seen in the previous section what the type
:class:`DataSubset` is, and why it exists. We also mentioned that
an end-user does not usually need to work with the constructor
:func:`DataSubset` directly. Instead, we recommended to always
just use :func:`datasubset` instead.

You may ask yourself right now why we were using this
:class:`DataSubset` type in the first place. After all, we saw
that calling the function :func:`datasubset` gave us a more
convenient ``SubArray`` to work with. Well, as we hinted before,
not every data container can be expected to be a subtype of
``AbstractArray``. To get a better understanding of why we care
about this, let us together explore the implications on a couple
of commonly used data sources that are available in the Julia
package ecosystem.

.. _dataframe:

Example: DataFrames.jl
~~~~~~~~~~~~~~~~~~~~~~~

Let's consider a type of data source that is very different to an
``Array``; a ``DataFrame`` from the `DataFrames.jl
<https://github.com/JuliaStats/DataFrames.jl>`_ package. By
default, a ``DataFrame`` is not a data container, because it does
not implement the required interface. We can change that however.

.. code-block:: jlcon

   julia> using DataFrames, LearnBase

   julia> LearnBase.getobs(df::DataFrame, idx) = df[idx,:]

   julia> LearnBase.nobs(df::DataFrame) = nrow(df)

With those two methods defined, every ``DataFrame`` is a fully
qualified data container. This means that it can now be
subsetted.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(4), x2 = rand(4))
   4×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> mysubset = datasubset(df, [2,4])
   MLDataPattern.DataSubset{DataFrames.DataFrame,Array{Int64,1},LearnBase.ObsDim.Undefined}
    2 observations

   julia> getobs(mysubset)
   2×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │
   │ 2   │ 0.522172 │ 0.722906  │

Notice how we used :func:`datasubset` here, instead of invoking
the :func:`DataSubset` constructor directly. This is the
recommended way of creating data subsets. The main difference is,
that :func:`datasubset` will try to choose the most appropriate
type to represent a subset for the given container, while the
constructor will always use :class:`DataSubset`.

.. _datatable:

Example: DataTables.jl
~~~~~~~~~~~~~~~~~~~~~~~~

Another good example for a custom data source are ``DataTable``
from the `DataTables.jl
<https://github.com/JuliaData/DataTables.jl>`_ package. This
rather new, table-like type is advertised as the "future of
working with tabular data in Julia". An interesting difference to
``DataFrame`` is that it offers a view-type called
``SubDataTable``, which is a perfect candidate for a custom data
subset type.

Not unlike ``DataFrame``, a ``DataTable`` is by default not a
data container, because it does not implement the required
interface. We will again change that. In contrast to before,
however, we will also implement a custom method for
:func:`datasubset`.

.. code-block:: jlcon

   julia> using DataTables, LearnBase

   julia> LearnBase.nobs(dt::AbstractDataTable) = nrow(dt)

   julia> LearnBase.getobs(dt::AbstractDataTable, idx) = dt[idx,:]

   julia> LearnBase.datasubset(dt::AbstractDataTable, idx, ::ObsDim.Undefined) = view(dt, idx)

..   julia> LearnBase.datasubset(dt::SubDataTable, idx) = view(dt.parent, dt.rows[idx])

It is worth pointing out that it is a current limitation that any
custom method for :func:`datasubset` must also include the third
parameter ``obsdim`` (even if it is undefined).

Now that we have the required interface implemented, every
``DataTable`` is regarded as a fully qualified data container. In
contrast to the ``DataFrame`` example, it even has its own
custom type for representing a data subset.

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(4), x2 = rand(4))
   4×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> mysubset = datasubset(dt, [2, 4])
   2×2 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │
   │ 2   │ 0.522172 │ 0.722906  │

   julia> datasubset(mysubset, 2) # subsetting a subset
   1×2 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1       │ x2       │
   ├─────┼──────────┼──────────┤
   │ 1   │ 0.522172 │ 0.722906 │

   julia> getobs(mysubset)
   2×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │
   │ 2   │ 0.522172 │ 0.722906  │

One may ask why we go through this trouble, if we could just use
``Base.view`` instead. Aside from the observation dimension
aspect when working with arrays, there are good reason for having
such a neutral interface. After all, a data subset is just a
means to an end. We will see in the following sections how
higher-level functions can create various data subsets in much
more useful ways than us just calling :func:`datasubset`
ourselves. So once some data source supports the data container
interface, all the high-level functionality that we will spend
the rest of this document on, comes with it for free.

Shuffling a Data Container
---------------------------

A vastly under-appreciated duty of any Machine Learning framework
is shuffling a data set (or parts of a data set). Shuffling the
order of the observations before training a model on that data
set is important for various practical and well known reasons. We
still call it under-appreciated, however, because it is easy to
implement "shuffling" inefficiently. That in turn can influence a
lot of dependent functionality; especially if big data sets are
involved. For example, it is not unusual that the shuffling is
performed very early in the ML pipeline. Depending on the design
of the framework, this could cause a lot of unnecessary data
movement.

In this package we follow the simple idea, that the "shuffling"
of a data set should be performed on an indices level, and not an
observation level. What that means is that instead of copying or
mutating the actual data, we simply create a lazy "subset" of
that data using shuffled indices. As a consequence, the actual
data remains untouched by the process until :func:`getobs` is
called. In other words, while the resulting subset points to the
same observations, it has the order of the indices shuffled. The
function that implements this functionality is called
:func:`shuffleobs`.

.. function:: shuffleobs(data, [obsdim])

   Return a "subset" of `data` that spans the same exact
   observations, but has the order of those observations
   permuted.

   The values of `data` itself are not copied. Instead only the
   indices are shuffled. This function calls :func:`datasubset`
   to accomplish that, which means that the return value is
   likely of a different type than data.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

This is where we will start to see the subtle beauty of the
package design. We have previously discussed in some detail how
to interact (and subset) data containers such as ``Array``,
``DataTable``, and ``DataFrame``. Let us now take a look at what
it means to "shuffle" each of those. First, we will consider a
plain Julia ``Array``.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> X_shuf = shuffleobs(X) # each column is an observation
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.933372  0.505208   0.0443222  0.226582
    0.522172  0.0997825  0.722906   0.504629

   julia> getobs(X_shuf) # copy into a Array
   2×4 Array{Float64,2}:
    0.933372  0.505208   0.0443222  0.226582
    0.522172  0.0997825  0.722906   0.504629

   julia> shuffleobs(X, obsdim = 1) # each row is an observation
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}:
    0.504629  0.522172  0.0997825  0.722906
    0.226582  0.933372  0.505208   0.0443222

As we can see, :func:`shuffleobs` returns a ``SubArray`` instead
of an ``Array``. As such, it still points at the data in ``X``.
To get the actual data as a proper ``Array`` (e.g. for memory
locality) we can use :func:`getobs` on the result. Also note how
the result of :func:`shuffleobs` depends on the specified
``obsdim``. This is because we just want to permute the order of
the observations, not the features.

Next we will take a look at what happens when we call
:func:`shuffleobs` with a ``DataTable``. Note that for this to
work it is required that the data container interface is
implemented (which we did as an exercise in :ref:`datatable`)

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(4), x2 = rand(4))
   4×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> dt_shuf = shuffleobs(dt)
   4×2 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.933372 │ 0.0443222 │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.226582 │ 0.505208  │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> getobs(dt_shuf)
   4×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.933372 │ 0.0443222 │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.226582 │ 0.505208  │
   │ 4   │ 0.522172 │ 0.722906  │

Note how the actual code did not change much, even though
``DataTables`` are quite different to ``Array``. We can again
observe how :func:`shuffleobs` did not return a new
``DataTable``, but instead a lazy view in the form of a
``SubDataTable``.

To mix it up a little, let us take a look at a data container
that does not provide its own type of data subset; a
``DataFrame``. Note that for the following code to work, it is
required that the data container interface is implemented (which
we did as an exercise in :ref:`dataframe`)

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(4), x2 = rand(4))
   4×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> df_shuf = shuffleobs(df)
   MLDataPattern.DataSubset{DataFrames.DataFrame,Array{Int64,1},LearnBase.ObsDim.Undefined}
    4 observations

   julia> getobs(df_shuf)
   4×2 DataFrames.DataFrame
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.933372 │ 0.0443222 │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.226582 │ 0.505208  │
   │ 4   │ 0.522172 │ 0.722906  │

Admittedly, the result of :func:`shuffleobs` does not look as
intuitive or information in this example. It does however do its
job perfectly, which is avoiding data access. This property of
:class:`DataSubset` is particularly useful if our data container
is some interface to a big remote data set. In such a case we
would like to avoid loading any data until we really need it.

Aside from a common interface for different data types, the real
power of using :func:`shuffleobs` is in linking multiple data
containers together on an per-observation level. This way they
can be shuffled as one coherent unit. To do that, simply put all
the relevant data container into a single ``Tuple``, before
passing it to :func:`shuffleobs`. For example, let's say that our
features are contained in a ``DataTable`` and the targets stored
in a separate ``Vector``.

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(4), x2 = rand(4))
   4×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> y = rand(4)
   4-element Array{Float64,1}:
    0.812814
    0.245457
    0.11202
    0.000341996

   julia> df_shuf, y_shuf = shuffleobs((dt, y))
   (4×2 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.504629 │ 0.0997825 │
   │ 2   │ 0.933372 │ 0.0443222 │
   │ 3   │ 0.522172 │ 0.722906  │
   │ 4   │ 0.226582 │ 0.505208  │,[0.245457,0.11202,0.000341996,0.812814])

   julia> typeof(y_shuf)
   SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}

As we can see, the observations in ``dt`` and ``y`` are both
shuffled in the same manner. Thus the per-observation link is
preserved and we can continue to treat it as a single data set.

Splitting into Train and Test
------------------------------

Some data preparation tasks, such as partitioning the data set
into a training-, (validation-,) and test-set, are often
performed offline or sometimes even predefined by a third party
(e.g. the initial authors of a benchmark data set). That said, it
is useful to efficiently and conveniently be able to split a
given data set into differently sized subsets.

For that purpose, this package provides a function called
:func:`splitobs`. As the name subtly hints, this function does
not shuffle the content, but instead performs a static split at
the relative position specified in ``at``.

.. function:: splitobs(data, [at = 0.7], [obsdim]) -> Tuple

   Split the given `data` into two disjoint subsets and returns
   them as a ``Tuple``. The first subset contains the fraction
   `at` of observations in `data`, and the second subset contains
   the rest.

   Note that this function will perform the splits statically and
   thus not perform any randomization.  If you want to perform a
   random assignment of observations to subset, use the function
   in combination with :func:`shuffleobs`.

   :param data: The object representing a data container.

   :param AbstractFloat at: \
        Optional. The fraction of observations that should be in
        the first subset. Must be in the interval (0,1). Can be
        specified as positional or keyword argument. Defaults to
        0.7 (i.e. 70% of the observations in the first subset).

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Let's consider an example feature matrix ``X`` in the form of an
``Array``, which has 8 observations with 2 features each.

.. code-block:: jlcon

   julia> X = rand(2, 8)
   2×8 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202      0.380001  0.841177
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996  0.505277  0.326561

We can split this data container into two subsets by calling
:func:`splitobs` with the desired relative split point. If ``at``
is specified as a floating point number, then the return-value
will be a ``Tuple`` with two elements (i.e. subsets), in which
the first subset contains the fraction of observations specified
by ``at`` and the second subset contains the rest. In the
following code the first subset ``train`` will contain the first
60% of the observations and the second subset ``test`` the rest.

.. code-block:: jlcon

   julia> train, test = splitobs(X, at = 0.6); # or splitobs(X, 0.6)

   julia> train
   2×5 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> test
   2×3 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.11202      0.380001  0.841177
    0.000341996  0.505277  0.326561

Note that we can provide the split point ``at`` as either a
type-stable positional argument, or as a more descriptive keyword
argument. Furthermore, it is worth to again point out that
:func:`splitobs` works for any type that implements the data
container interface (see :ref:`datatable` to make the following
code work).

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(4), x2 = rand(4))
   4×2 DataTables.DataTable
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │
   │ 4   │ 0.522172 │ 0.722906  │

   julia> train, test = splitobs(dt, at = 0.8);

   julia> train
   3×2 DataTables.SubDataTable{UnitRange{Int64}}
   │ Row │ x1       │ x2        │
   ├─────┼──────────┼───────────┤
   │ 1   │ 0.226582 │ 0.505208  │
   │ 2   │ 0.504629 │ 0.0997825 │
   │ 3   │ 0.933372 │ 0.0443222 │

   julia> test
   1×2 DataTables.SubDataTable{UnitRange{Int64}}
   │ Row │ x1       │ x2       │
   ├─────┼──────────┼──────────┤
   │ 1   │ 0.522172 │ 0.722906 │

Naturally, :func:`splitobs` also supports the optional parameter
``obsdim``, which is especially useful for arrays. It can be
specified as either a positional argument, or as a keyword
argument. See :ref:`obsdim` for more information.

.. code-block:: jlcon

   julia> train, test = splitobs(X', at = 0.6); # note the transpose

   julia> train
   5×2 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}:
    0.226582   0.504629
    0.933372   0.522172
    0.505208   0.0997825
    0.0443222  0.722906
    0.812814   0.245457

   julia> test
   3×2 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}:
    0.11202   0.000341996
    0.380001  0.505277
    0.841177  0.326561

It is also possible to call :func:`splitobs` with multiple data
container wrapped in a ``Tuple``, which all must have the same
number of total observations. This will link the data containers
together on a per-observation basis and is especially useful for
labeled data.

.. code-block:: jlcon

   julia> X = rand(2,8)
   2×8 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202      0.380001  0.841177
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996  0.505277  0.326561

   julia> y = rand(8)
   8-element Array{Float64,1}:
    0.810857
    0.850456
    0.478053
    0.179066
    0.44701
    0.219519
    0.677372
    0.746407

   julia> train, test = splitobs((X, y), at = 0.6); # train and test are both a Tuple

   julia> (x_train,y_train), (x_test,y_test) = splitobs((X, y), at = 0.6); # same but splat

   julia> x_train
   2×5 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> y_train
   5-element SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}:
    0.810857
    0.850456
    0.478053
    0.179066
    0.44701

   julia> x_test
   2×3 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.11202      0.380001  0.841177
    0.000341996  0.505277  0.326561

   julia> y_test
   3-element SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}:
    0.219519
    0.677372
    0.746407

As we can see, the function performs a static split and not a
random assignment. This may not always be what we really want.
For that purpose, this package provides a function called
:func:`shuffleobs`, which we introduced in an earlier section.
Using :func:`shuffleobs` in combination with :func:`splitobs`
will result in a random assignment of observations to the data
partitions.

.. code-block:: jlcon

   julia> train, test = splitobs(shuffleobs(X), at = 0.6);

   julia> train
   2×5 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.841177  0.812814  0.226582  0.11202      0.933372
    0.326561  0.245457  0.504629  0.000341996  0.522172

   julia> test
   2×3 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.0443222  0.380001  0.505208
    0.722906   0.505277  0.0997825

So far we have only seen how to partition one or more data
container into exactly two disjoint data subsets. The function
:func:`splitobs` allows for an arbitrary amount of partitions
however. To create :math:`N` partitions, you simply need to
specify a tuple of :math:`N-1` fractions. The sum of all
fractions must be in the interval (0,1).

.. function:: splitobs(data, at, [obsdim]) -> NTuple

   Split the given `data` into multiple disjoint subsets with
   sizes proportional to the value(s) of `at`.

   Note that this function will perform the splits statically and
   thus not perform any randomization. The function creates a
   ``NTuple`` of data subsets in which the first :math:`N-1`
   elements/subsets contain the fraction of observations from
   `data` that is specified by the values in `at`. The last tuple
   element will then contain the rest of the data.

   :param data: The object representing a data container.

   :param Tuple at: \
        Tuple of fractions. All elements must be positive and
        their sum must be in the interval (0,1).

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Creating more than two data partitions is particularly convenient
for creating an additional validation set. In the following
example ``train`` will have the first 50% of the observations,
``val`` will have next 40%, and ``test`` the last 10%

.. code-block:: jlcon

   julia> X = rand(2,8)
   2×8 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202      0.380001  0.841177
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996  0.505277  0.326561

   julia> train, val, test = splitobs(X, at = (0.5, 0.4));

   julia> train
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> val
   2×3 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.812814  0.11202      0.380001
    0.245457  0.000341996  0.505277

   julia> test
   2×1 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.841177
    0.326561
