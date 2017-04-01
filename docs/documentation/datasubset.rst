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
to avoid copying our data set around until we actually need it.
To that end, we provide the function :func:`datasubset`, which
tries to choose the most appropriate type of subset for the given
data container.

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
        `data` that should be returned. Can be of type ``Int`` or
        some subtype ``AbstractVector{Int}``. Defaults to
        ``1:nobs(data,obsdim)``

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
arrays, we represent every subset of an arrays as ``SubArray``.

.. code-block:: jlcon

   julia> X = rand(2,4)
   2×4 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222
    0.504629  0.522172  0.0997825  0.722906

   julia> datasubset(X, 2) # single observation
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
explicitly specifying the ``obsdim``.

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
container ``X`` that uses the accumulated indices. In other
words, while subsetting operations can be nested, they should be
combined into a single layer (i.e. you don't want a subset of a
subset of a subset represented as nested types)

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
tuple-order.

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

So far, we have only considered subsetting data of type
``Array``. However, what if we want to subset some other data
container that does not implement the ``AbstractArray``
interface? Naturally, we can't just use ``SubArray`` to represent
those subsets. For that purpose we provide a generic type
:class:`DataSubset`, that serves as the default subset type for
every data container that does not implement their own methods
for :func:`datasubset`.

.. class:: DataSubset

   Used as the default type to represent a subset of some
   arbitrary data container. Its main task is keeping track of
   which observation-indices the subset spans. As such it is
   designed in a way, that makes sure that subsequent subsettings
   are accumulated without needing to access actual data.

   The main purpose for the existence of ``DataSubset`` is to
   delay data-access and -movement until an actual batch of data
   (or single observation) is needed for some computation. This
   is particularily useful when the data is not located in
   memory, but on the hard drive or some remote location. In such
   a scenario one wants to load the required data only when
   needed.

.. function:: DataSubset(data, [idx], [obsdim]) -> DataSubset

   Create an instance of ``DataSubset`` that will represent a
   lazy subset of the observations in `data` corresponding to the
   given index/indices in `idx`. No data will be copied except of
   the indices

   If `data` is a ``DataSubset``, then the indices of the subset
   will be combined with `idx` and a new, accumulated
   ``DataSubset`` will be returns.

   :param data: The object representing a data container.

   :param idx: \
        Optional. The index or indices of the observation(s) in
        `data` that should be returned. Can be of type ``Int`` or
        some subtype ``AbstractVector{Int}``. Defaults to
        ``1:nobs(data,obsdim)``

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

TODO: reason for existing.. fallback subset type

TODO: show that you can use it with arrays (as toy example)

TODO: example use with DataFrame

TODO: show that datasubset will trigger it


Shuffling a Data Container
---------------------------

TODO: splitobs definition with float

TODO: splitobs definition with tuple of floats

TODO: using tuple of data container

Splitting into Train and Test
------------------------------

Some separation strategies, such as dividing the data set into a
training- and a testset, is often performed offline or predefined
by a third party. That said, it is useful to efficiently and
conveniently be able to split a given data set into differently
sized subsets.

One such function that this package provides is called
:func:`splitobs`.  Note that this function does not shuffle the
content, but instead performs a static split at the relative
position specified in ``at``.

TODO: example splitobs

For the use-cases in which one wants to instead do a completely
random partitioning to create a training- and a testset, this
package provides a function called `shuffleobs`.  Returns a lazy
"subset" of data (using all observations), with only the order of
the indices permuted. Aside from the indices themseves, this is
non-copy operation. Using :func:`shuffleobs` in combination with
:func:`splitobs` thus results in a random assignment of
data-points to the data-partitions.

TODO: example shuffleobs

K-Folds for Cross-validation
-----------------------------

Yet another use-case for data partitioning is model selection;
that is to determine what hyper-parameter values to use for a
given problem. A particularly popular method for that is *k-fold
cross-validation*, in which the data set gets partitioned into
:math:`k` folds. Each model is fit :math:`k` times, while each
time a different fold is left out during training, and is instead
used as a validation set. The performance of the :math:`k`
instances of the model is then averaged over all folds and
reported as the performance for the particular set of
hyper-parameters.

This package offers a general abstraction to perform
:math:`k`-fold partitioning on data sets of arbitrary type. In
other words, the purpose of the type :class:`KFolds` is to provide
an abstraction to randomly partition some data set into :math:`k`
disjoint folds. :class:`KFolds` is best utilized as an iterator.
If used as such, the data set will be split into different
training and test portions in :math:`k` different and unqiue
ways, each time using a different fold as the validation/testset.

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.

