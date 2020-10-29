.. _labeledcontainer:

Labeled Data Container
=======================

Depending on the domain-specific problem and the data one is
working with, a data source may or may not contain what is known
as **targets**. A target (singular) is a piece of information
about a single observation that represents the desired output (or
correct answer) for that specific observation. If targets are
involved, then we find ourselves in the realm of supervised
learning. This includes both, classification (predicting
categorical output) and regression (predicting real-valued
output).

Dealing with targets in a generic way was quite a design
challenge for us. There are a lot of aspects to consider and
issues to address in order to achieve an extensible and flexible
package architecture, that can deal with labeled- and unlabeled
data sources equally well.

- Some data container may not contain any targets or even
  understand what targets are.

- There are labeled data sources that are not considered
  :ref:`container` (like many data iterators). A flexible package
  design needs a reasonably consistent API for both cases.

- The targets can be in a different data container than the
  features. For example it is quite common to store the features
  in a matrix :math:`X`, while the corresponding targets
  are stored in a separate vector :math:`\vec{y}`.

- For some data container, the targets are an intrinsic part of
  the observations. Furthermore, it might be the case that every
  data set has its own convention concerning which part of an
  observation represents the target. An example for such a data
  container is a ``DataFrame`` where one column denotes the
  target. The name/index of the target-column depends on the
  concrete data set, and is in general different for each
  ``DataFrame``. In other words, this means that for some data
  containers, the type itself does not know how to access a
  target. Instead it has to be a user decision.

- There are scenarios, where a data container just serves as an
  interface to some remote data set, or a big data set that is
  stored on the disk. If so, it is likely the case, that the
  targets are not part of the observations, but instead part of
  the data container metadata. An example would be a data
  container that represents a nested directory of images in the
  file system. Each sub-directory would then contain all the
  images of a single class. In that scenario, the targets are
  known from the directory names and could be part of some
  metadata. As such, it would be far more efficient if the data
  container can make use of this information, instead of having
  to load an actual image from the disk just to access its
  target. Remember that targets are not only needed during
  training itself, but also for data partitioning and resampling.

The implemented target logic is in some ways a bit more complex
than the :func:`getobs` logic. The main reason for this is that
while :func:`getobs` is solely designed for data containers, we
want the target logic to seamlessly support a wide variety of
data sources and data scenarios. In this document, however, we
will only focus on data sources that are considered labeled
:ref:`container`. Note that in the context of this package, a
**labeled data container** is a data container that contains
*targets*; be it categorical or continuous.

Query Target(s)
-------------------

The first question one may ask is: "Why would the access pattern
need to *extract* the targets out of some data container?". After
all, it would be simpler to just pass the targets as an
additional parameter to any function that needs them. In fact,
that is pretty much how almost all other ML frameworks handle
labeled data. The reason why we diverge from this tradition is
two-fold.

1. The set of access pattern that work on labeled data is really
   just a superset of the set of access pattern that work on
   unlabeled data. So by doing it our way, we avoid duplicate
   code.

2. The second (and more important) reason is that we decided that
   there is really no convincing argument for restricting the
   user input to either be in the form of one variable (unlabeled
   data), or two variables (for labeled data). In fact, we wanted
   to allow the same variable to contain the features as well as
   targets. We also wanted to allow users to work with mutliple
   data sources that don't contain any targets at all.

To that end we provide the function :func:`targets`. It can be
used to query all the, well, targets of some given labeled data
container or data subset.

.. function:: targets(data, [obsdim])

   Query the concrete targets from `data` and return them.

   This function is eager in the sense that it will always call
   :func:`getobs` unless a custom method for
   ``LearnBase.gettargets`` (see later) is implemented for the
   type of `data`. This will make sure that actual values are
   returned (in contrast to placeholders such as
   :class:`DataSubset` or ``SubArray``).

   In other words, the returned values must be in the form
   intended to be passed as-is to some resampling strategy or
   learning algorithm.

   :param data: The object representing a labeled data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

In some cases we will see that invoking :func:`targets` just
seems to return the given data container unchanged. The reason
for this is simple. What :func:`targets` tries to do is return
the portion of the given data container that corresponds to the
targets. The function assumes that there *must* be targets of
some sorts (otherwise, why would you call a function called
"targets"?). If there is no decision to be made (e.g. there is
only a single vector to begin with), then the function simply
returns the result of :func:`getobs` for the given data.

.. code-block:: jlcon

   julia> targets([1,2,3,4])
   4-element Array{Int64,1}:
    1
    2
    3
    4

The above edge-case isn't really that informative for the main
functionality that :func:`targets` provides. The more interesting
behaviour can be seen for custom types and/or tuples. More
specifically, if the given data is a ``Tuple``, then the
convention is that the last element of the tuple contains the
targets and the function is recursed once (and only once).

.. code-block:: jlcon

   julia> targets(([1,2], [3,4]))
   2-element Array{Int64,1}:
    3
    4

   julia> targets(([1,2], ([3,4], [5,6])))
   ([3,4],[5,6])

What this shows us is that we can use tuples to create a labeled
data container out of two simple data containers. This is
particularly useful when working with arrays. Considering the
following situation, where we have a feature matrix ``X`` and a
corresponding target vector ``y``.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.987618  0.365172  0.306373  0.540434  0.805117
    0.801862  0.469959  0.704691  0.405842  0.014829

   julia> y = [:a, :a, :b, :a, :b]
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

   julia> targets((X, y))
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

You may have noticed from the signature of :func:`targets`, that
there is no parameter for passing indices. This is no accident.
The purpose of :func:`targets` is not subsetting, it is to
extract the targets; no more, no less. If you wish to only query
the targets of a subset of some data container, you can use
:func:`targets` in combination with :func:`datasubset`.

.. code-block:: jlcon

   julia> targets(datasubset((X, y), 2:3))
   2-element Array{Symbol,1}:
    :a
    :b

If the type of the data itself is not sufficient information to
be able to extract the targets, one can specify a
target-extraction-function ``fun`` that is to be applied to each
observation. This function must be passed as the first parameter
to :func:`targets`.

.. function:: targets(fun, data, [obsdim]) -> Vector

   Extract the concrete targets from the observations in `data`
   by applying `fun` on each observation individually. The
   extracted targets are returned as a ``Vector``, which
   preserves the order of the observations from `data`.

   :param fun: \
        A callable object (usually a function) that should
        be applied to each observation individually in order to
        extract or compute the target for that observation.

   :param data: The object representing a labeled data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

A great example for a data source, that stores the features and
the targets in the same manner, is a ``DataFrame``. There is no
clear convention what column of the table denotes the targets; it
depends on the data set. As such, we require a data-specific
target-extraction-function. Consider the following example using
a toy ``DataFrame`` (see :ref:`dataframe` to make the following
code work). For this particular data frame we know that the
column ``:y`` contains the targets.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])
   5×3 DataFrames.DataFrame
   │ Row │ x1        │ x2       │ y │
   ├─────┼───────────┼──────────┼───┤
   │ 1   │ 0.176654  │ 0.821837 │ a │
   │ 2   │ 0.0397664 │ 0.894399 │ a │
   │ 3   │ 0.390938  │ 0.29062  │ b │
   │ 4   │ 0.582912  │ 0.509047 │ a │
   │ 5   │ 0.407289  │ 0.113006 │ b │

   julia> targets(row->row.y, df)
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

Another use-case for specifying an extraction function, is to
discretize some continuous regression targets. We will see later,
when we start discussing higher-level functions, how this can be
useful in order to over- or under-sample the data set (see
:func:`oversample` or :func:`undersample`).

.. code-block:: jlcon

   julia> targets(x -> (x > 0.7), rand(6))
   6-element Array{Bool,1}:
     true
    false
     true
    false
     true
     true

Note that if this optional first parameter (i.e. the extraction
function) is passed to :func:`targets`, it will always be applied
to the observations, and **not** the container. In other words,
the first parameter is applied to each observation individually
and not to the data as a whole. In general this means that the
return type changes drastically, even if passing a no-op
function.

.. code-block:: jlcon

   julia> X = rand(2, 3)
   2×3 Array{Float64,2}:
    0.105307   0.58033   0.724643
    0.0433558  0.116124  0.89431

   julia> y = [1 3 5; 2 4 6]
   2×3 Array{Int64,2}:
    1  3  5
    2  4  6

   julia> targets((X,y))
   2×3 Array{Int64,2}:
    1  3  5
    2  4  6

   julia> targets(x->x, (X,y))
   3-element Array{Array{Int64,1},1}:
    [1,2]
    [3,4]
    [5,6]

We can see in the above example, that the default assumption for
an ``Array`` of higher order is that the last array dimension
enumerates the observations. The optional parameter ``obsdim``
can be used to explicitly overwrite that default. If the concept
of an observation dimension is not defined for the type of
``data``, then ``obsdim`` can simply be omitted.

.. code-block:: jlcon

   julia> X = [1 0; 0 1; 1 0]
   3×2 Array{Int64,2}:
    1  0
    0  1
    1  0

   julia> targets(indmax, X, obsdim=1)
   3-element Array{Int64,1}:
    1
    2
    1

   julia> targets(indmax, X, ObsDim.First())
   3-element Array{Int64,1}:
    1
    2
    1

Note how ``obsdim`` can either be provided using type-stable
positional arguments from the namespace ``ObsDim``, or by using a
more flexible and convenient keyword argument. See :ref:`obsdim`
for more information on that topic.

Iterate over Targets
---------------------

In some situations, one only wants to *iterate* over the targets,
instead of querying all of them at once. In those scenarios it
would be beneficial to avoid the allocation temporary memory all
together. To that end we provide the function :func:`eachtarget`,
which returns a ``Base.Generator``.

.. function:: eachtarget([fun], data, [obsdim]) -> Generator

   Return a ``Base.Generator`` that iterates over all targets in
   `data` once and in the right order. If `fun` is provided it
   will be applied to each observation in data.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the target for that
        observation.

   :param data: The object representing a labeled data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

The function :func:`eachtarget` behaves very similar to
:func:`targets`. For example, if you pass it a ``Tuple`` of data
container, then it will assume that the last tuple element
contains the targets.

.. code-block:: jlcon

   julia> iter = eachtarget(([1,2], [3,4]))
   Base.Generator{UnitRange{Int64},MLDataPattern.##79#80{2,Tuple{Array{Int64,1},Array{Int64,1}},Tuple{LearnBase.ObsDim.Last,LearnBase.ObsDim.Last}}}(MLDataPattern.#79,1:2)

   julia> collect(iter)
   2-element Array{Int64,1}:
    3
    4

The one big difference to :func:`targets` is that
:func:`eachtarget` will always iterate over the targets one
observation at a time, regardless whether or not an extraction
function is provided.

.. code-block:: jlcon

   julia> iter = eachtarget([1 3 5; 2 4 6])
   Base.Generator{UnitRange{Int64},MLDataPattern.##72#73{Array{Int64,2},LearnBase.ObsDim.Last}}(MLDataPattern.#72,1:3)

   julia> collect(iter)
   3-element Array{Array{Int64,1},1}:
    [1,2]
    [3,4]
    [5,6]

   julia> targets([1 3 5; 2 4 6]) # as comparison
   2×3 Array{Int64,2}:
    1  3  5
    2  4  6

Of course, it is also possible to work with any other type of
data source that is considered a :ref:`container`. Consider the
following example using a toy ``DataFrame`` (see :ref:`dataframe`
to make the following code work). For this particular data frame
we will assume that the column ``:y`` contains the targets.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])
   5×3 DataFrames.DataFrame
   │ Row │ x1        │ x2       │ y │
   ├─────┼───────────┼──────────┼───┤
   │ 1   │ 0.176654  │ 0.821837 │ a │
   │ 2   │ 0.0397664 │ 0.894399 │ a │
   │ 3   │ 0.390938  │ 0.29062  │ b │
   │ 4   │ 0.582912  │ 0.509047 │ a │
   │ 5   │ 0.407289  │ 0.113006 │ b │

   julia> iter = eachtarget(row->row.y, df)
   Base.Generator{MLDataPattern.ObsView{MLDataPattern.DataSubset{DataFrames.DataFrame,Int64,LearnBase.ObsDim.Undefined},...

   julia> collect(iter)
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

Just like for :func:`targets`, the optional parameter ``obsdim``
can be used to specify which dimension denotes the observations,
if that concept makes sense for the type of the given data.

.. code-block:: jlcon

   julia> X = [1 0; 0 1; 1 0]
   3×2 Array{Int64,2}:
    1  0
    0  1
    1  0

   julia> iter = eachtarget(indmax, X, obsdim = 1)
   Base.Generator{MLDataPattern.ObsView{SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Colon},true},Array{Int64,2},LearnBase.ObsDim.Constant{1}},...

   julia> collect(iter)
   3-element Array{Int64,1}:
    1
    2
    1

.. _customtargets:

Support for Custom Types
--------------------------

Any labeled data container has the option to customize the
behaviour of :func:`targets`. The emphasis here is on "option",
because it is not required by the interface itself. Aside from
leaving the default behaviour, there are two ways to customize
the logic behind :func:`targets`.

1. Implement ``LearnBase.gettargets`` for the **data container**
   type. This will bypasses the function :func:`getobs` entirely,
   which can significantly improve the performance.

2. Implement ``LearnBase.gettarget`` for the **observation**
   type, which is applied on the result of :func:`getobs`. This
   is useful when the observation itself contains the target.

Let us consider two example scenarios that benefit from
implementing custom methods. The first one for
``LearnBase.gettargets``, and the second one for
``LearnBase.gettarget``. Note again that these functions are
internal and only intended to be *extended* by the user (and
**not** called). A user should not use them directly but instead
always call :func:`targets` or :func:`eachtarget`.

Example 1: Custom File-Based Data Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you want to write a custom data container that
describes a directory on your hard-drive. Each sub-directory is
expected to contain a set of large images that belong to a single
class (the directory name). This kind of data container only
loads the images itself if they are actually needed (so on
:func:`getobs`). The targets, however, would technically be
available in the memory at all times, since it is part of the
metadata.

To "simulate" such a scenario, let us define a dummy type that
represents the idea of such a data container for which each
observation is expensive to access, but where the corresponding
targets are available in some member variable.

.. code-block:: julia

   using LearnBase

   immutable DummyDirImageSource
       targets::Vector{String}
   end

   LearnBase.getobs(::DummyDirImageSource, i) = error("expensive computation triggered")

   StatsBase.nobs(data::DummyDirImageSource) = length(data.targets)

Naturally, we would like to avoid calling :func:`getobs` if at
all possible. While we can't avoid calling :func:`getobs` when we
actually need the data, we could avoid it when we only require
the targets (for example for data partitioning or resampling).
This is because in this example, the targets are part of the
metadata that is always loaded. We can make use of this fact by
implementing a custom method for ``LearnBase.gettargets``.

.. code-block:: julia

   LearnBase.gettargets(data::DummyDirImageSource, i) = data.targets[i]

By defining this method, the function :func:`targets` can now
query the targets efficiently by looking them up in the member
variable. In other words it allows to provide the targets of some
observation(s) without ever calling :func:`getobs`. This even
works seamlessly in combination with :func:`datasubset`.

.. code-block:: jlcon

   julia> source = DummyDirImageSource(["malign", "benign", "benign", "malign", "benign"])
   DummyDirImageSource(String["malign","benign","benign","malign","benign"])

   julia> targets(source)
   5-element Array{String,1}:
    "malign"
    "benign"
    "benign"
    "malign"
    "benign"

   julia> targets(datasubset(source, 3:4))
   2-element Array{String,1}:
    "benign"
    "malign"

Note however, that calling :func:`targets` with a
target-extraction-function will still trigger :func:`getobs`.
This is expected behaviour, since the extraction function is
intended to "extract" the target from each actual observation
(i.e. the result of :func:`getobs`).

.. code-block:: jlcon

   julia> targets(x->x, source)
   ERROR: expensive computation triggered

Example 2: Symbol Support for DataFrames.jl
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DataFrame`` are a kind of data container for which the targets
are as much part of the data as the features are (in contrast to
Example 1). Furthermore, each observation is itself also a
``DataFrame``. Before we start, let us implement the required
:ref:`container` interface.

.. code-block:: julia

   using DataFrames, LearnBase

   LearnBase.getobs(df::DataFrame, idx) = df[idx,:]

   StatsBase.nobs(df::DataFrame) = nrow(df)

Here we are fine with :func:`getobs` being called, since we need
to access the actual ``DataFrame`` anyway. However, we still need
to specify which column actually describes the features. This can
be done generically by specifying a target-extraction-function.

.. code-block:: jlcon

   julia> df = DataFrame(x1 = rand(5), x2 = rand(5), y = [:a,:a,:b,:a,:b])
   5×3 DataFrames.DataFrame
   │ Row │ x1       │ x2        │ y │
   ├─────┼──────────┼───────────┼───┤
   │ 1   │ 0.226582 │ 0.0997825 │ a │
   │ 2   │ 0.504629 │ 0.0443222 │ a │
   │ 3   │ 0.933372 │ 0.722906  │ b │
   │ 4   │ 0.522172 │ 0.812814  │ a │
   │ 5   │ 0.505208 │ 0.245457  │ b │

   julia> targets(row->row.y, df)
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

Alternatively, we could also implement a convenience syntax by
overloading ``LearnBase.gettarget``.

.. code-block:: julia

   LearnBase.gettarget(col::Symbol, df::DataFrameRow) = df[col]

This now allows us to call ``targets(:y, dataframe)``. While not
strictly necessary in this case, it can be quite useful for
special types of observations, such as ``ImageMeta``.

.. code-block:: jlcon

   julia> targets(:y, df)
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

We could even implement the default assumption, that the last
column denotes the targets unless otherwise specified.

.. code-block:: julia

   LearnBase.gettarget(df::DataFrameRow) = df[end]

Note that this might not be a good idea for a ``DataFrame`` in
particular. The purpose of this exercise is solely to show what
is possible.

.. code-block:: jlcon

   julia> targets(df)
   5-element Array{Symbol,1}:
    :a
    :a
    :b
    :a
    :b

.. _stratified:

Stratified Sampling
--------------------------

In a supervised learning scenario, in which we are usually
confronted with a labeled data set, we have to be considerate of
the distribution of the targets. That is, how likely is it to
observe some given target-value (e.g. an observation labeled
"malignant") without conditioning on the features.

It is important to be aware of the class distribution for a
couple of different reason, one of which is data partitioning.
Usually a good idea is to make sure that we actively try to
preserve the class distribution for every data subset. This will
help to make sure that the data subsets are similar in structure
and more likely to be representative of the full data set.

Consider the following target vector ``y``. Note how there are
only two elements of value ``:b``. If we just use random
assignment to partition the data set, then chances are that in
some cases one subset does not contain any element of value
``b``. This kind of effect becomes less frequent as the size of
the data set increases.

.. code-block:: jlcon

   julia> y = [:a, :a, :a, :a, :b, :b];

   julia> splitobs(shuffleobs(y), 0.5)
   (Symbol[:b,:b,:a],Symbol[:a,:a,:a])

To perform partitioning using stratified sampling without
replacement, this package provides the function
:func:`stratifiedobs`.

.. function:: stratifiedobs([fun], data, [p], [shuffle], [obsdim])

   Partition the data into multiple disjoint subsets proportional
   to the value(s) of `p`. The observations are assignmed to a
   data subset using stratified sampling without replacement.
   These subsets are then returned as a Tuple of subsets, where
   the first element contains the fraction of observations of
   data that is specified by the first float in `p`.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the label for that
        observation.

   :param data: The object representing a labeled data container.

   :param p: \
        Optional. The fraction of observations that should be in
        the first subset. Must be in the interval (0,1). Can be
        specified as positional or keyword argument, as either a
        single float or a tuple of floats. Defaults to 0.7 (i.e.
        70% of the observations in the first subset).

   :param bool shuffle: \
        Optional. Determines if the resulting data subsets will
        be shuffled after their creation. If ``false``, then all
        the observations will be clustered together accoring to
        their class label in each subset. Note that this has
        nothing to do with random assignment to some data subset,
        it only inluences the order of observation in each subset
        individually. Defaults to ``true``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Let's consider the following toy data vector ``y``, which
contains a total of 9 observations. Notice how each value of
``y`` is either ``:a`` or ``:b``, which means we have a binary
classification problem. We can also see that the data set has
twice as many observations for ``:a`` as it has for ``:b``.

.. code-block:: julia

   y = [:a, :a, :a, :a, :a, :a, :b, :b, :b]

We have seen before that using :func:`splitobs` can result in a
partition where one subset contains all the ``:b``, while the
other one contains only ``:a``. In contrast to this,
:func:`stratifiedobs` will try to make sure that both subsets are
both appropriately distributed. More concretely, if ``p`` is a
``Float64``, then the return-value of :func:`stratifiedobs` will
be a tuple with two elements (i.e. subsets), in which the first
element contains the fraction of observations specified by ``p``
and the second element contains the rest. In the following code
the first subset ``train`` will contain around 70% of the
observations and the second subset ``test`` the rest.

.. code-block:: jlcon

   julia> train, test = stratifiedobs(y, p = 0.7)
   (Symbol[:b,:a,:a,:a,:a,:b],Symbol[:a,:a,:b])

   julia> test
   3-element SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}:
    :a
    :a
    :b

Notice how both subsets contain twice as much ``:a`` as ``:b``,
just like ``y`` does. Furthermore, it is worth pointing out how
``test`` (and ``train`` for that matter) is a ``SubArray``. As
such, it is just a view into the corresponding original variable
``y``. The motivation for this behaviour is to avoid data
movement until :func:`getobs` is called.

Recall how I said explicitly that :func:`stratifiedobs` will "try
to make sure" that the distribution is preserved. This is because
it is not always possible to preserve the class distribution. If
that is the case, the function will try to have each subset
contain at least one of each class, even if that does not reflect
the distribution appropriately.

.. code-block:: jlcon

   julia> train, test = stratifiedobs(y, p = 0.8)
   (Symbol[:b,:a,:b,:a,:a,:a,:a],Symbol[:a,:b])

It is also possible to specify multiple fractions for ``p``. If
``p`` is a ``Tuple`` of ``Float64``, then additional subsets will
be created. This can be useful to create an additional validation
set.

.. code-block:: jlcon

   julia> train, val, test = stratifiedobs(y, p = (0.3, 0.3))
   (Symbol[:b,:a,:a],Symbol[:b,:a,:a],Symbol[:a,:b,:a])

It is also possible to call :func:`stratifiedobs` with multiple
data arguments as tuple, which all must have the same number of
total observations. Note that if data is a tuple, then it will be
assumed that the last element of the tuple contains the targets.

.. code-block:: julia

   train, test = stratifiedobs((X, y), p = 0.7)
   (X_train,y_train), (X_test,y_test) = stratifiedobs((X, y), p = 0.7)

If the type of the data is not sufficient information to be able
to extract the targets, one can specify a
target-extraction-function ``fun``, that is to be applied to each
individual observation. The behaviour when specifying or omitting
``fun`` is equivalent to its behaviour for :func:`eachtarget`,
because that is the function that is used internally by
:func:`stratifiedobs`.

A good example for a type that requires the parameter ``fun`` is
a ``DataTable``. Consider the following toy data container where
we know that the column ``:y`` contains the targets (see
:ref:`datatable` to make the following code work).

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(6), x2 = rand(6), y = [:a,:b,:b,:b,:b,:a])
   6×3 DataTables.DataTable
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.226582  │ 0.0443222   │ :a │
   │ 2   │ 0.504629  │ 0.722906    │ :b │
   │ 3   │ 0.933372  │ 0.812814    │ :b │
   │ 4   │ 0.522172  │ 0.245457    │ :b │
   │ 5   │ 0.505208  │ 0.11202     │ :b │
   │ 6   │ 0.0997825 │ 0.000341996 │ :a │

   julia> train, test = stratifiedobs(row->row[1,:y], dt)
   (3×3 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.505208  │ 0.11202     │ :b │
   │ 2   │ 0.522172  │ 0.245457    │ :b │
   │ 3   │ 0.0997825 │ 0.000341996 │ :a │,
   3×3 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1       │ x2        │ y  │
   ├─────┼──────────┼───────────┼────┤
   │ 1   │ 0.226582 │ 0.0443222 │ :a │
   │ 2   │ 0.933372 │ 0.812814  │ :b │
   │ 3   │ 0.504629 │ 0.722906  │ :b │)

The optional parameter ``obsdim`` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of data. For instance, consider the following toy
data set where the targets ``Y`` are now a one-of-k encoded
matrix. In such a case we would like the be able to re-sample
without first having to convert ``Y`` to a different
class-encoding.

.. code-block:: jlcon

   # 2 imbalanced classes in one-of-k encoding
   julia> Y = [1 0; 1 0; 1 0; 1 0; 0 1; 0 1]
   6×2 Array{Int64,2}:
    1  0
    1  0
    1  0
    1  0
    0  1
    0  1

Here we could use the function ``indmax`` to discretize the
individual target vectors on the fly. Remember that the
target-extraction-function is applied on each individual
observation in ``Y``. Since ``Y`` is a matrix, each observation
is a vector slice. Here we use ``obsdim`` to specify that each
row is an observation.

.. code-block:: jlcon

   julia> train, test = stratifiedobs(indmax, X, p = 0.5, obsdim = 1)
   ([1 0; 1 0; 0 1], [0 1; 1 0; 1 0])

.. _resampling:

Under- and Over-Sampling
---------------------------

It is not uncommon in a classification setting, that we find
ourselves working with an *imbalanced data set*. We call a
labeled data set **imbalanced**, if it contains more observations
of some class(es) than for the other(s). Training on such a data
set can pose a significant challenge for many commonly used
algorithms; especially if the difference in the class frequency
is large.

There are different conceptual approaches for dealing with
imbalanced data. A quite simple but popular strategy that works
for *data containers*, is to either under- or over-sample it
according to the class distribution. What that means is that the
data container is re-sampled in such a way, that the class
distribution in the resulting data container is approximately
uniform.

This package provides two functions to re-sample an imbalanced
data container; the first of which is called :func:`undersample`.
When under-sampling a data container, it will be down-sampled in
such a way, that each class has about as many observations in the
resulting subset, as the least represented class has in the
original data container.

.. function:: undersample([fun], data, [shuffle], [obsdim])

   Generate a class-balanced version of `data` by down-sampling
   its observations in such a way that the resulting number of
   observations will be the same number for every class. This
   way, all classes will have as many observations in the
   resulting data subset as the smallest class has in the given
   (i.e. original) `data` container.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the label for that
        observation.

   :param data: The object representing a labeled data container.

   :param bool shuffle: \
        Optional. Determines if the resulting data will be
        shuffled after its creation. If ``false``, then all the
        observations will be in their original order. Defaults to
        ``false``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: A down-sampled but class-balanced version of `data`
            in the form of a lazy data subset. No data is copied
            until :func:`getobs` is called.

Let's consider the following toy data set, which consists of a
feature matrix ``X`` and a corresponding target vector ``y``.
Both variables are data containers with 6 observations.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> y = ["a", "b", "b", "b", "b", "a"]
   6-element Array{String,1}:
    "a"
    "b"
    "b"
    "b"
    "b"
    "a"

As we can see, the target of each observation is either ``"a"``
or ``"b"``, which means we have a binary classification problem.
We can also see that the data set has twice as many observations
for ``"b"`` as it has for ``"a"``. Thus we consider it
imbalanced.

We can down-sample our toy data set by passing it to
:func:`undersample`. In order to tell the function that these two
data containers should be treated as a one data set, we have to
group them together using a ``Tuple``. This will cause
:func:`undersample` to return a ``Tuple`` of equal length and
ordering.

.. code-block:: jlcon

   julia> X_bal, y_bal = undersample((X, y));

   julia> X_bal
   2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.226582  0.933372  0.0443222  0.11202
    0.504629  0.522172  0.722906   0.000341996

   julia> y_bal
   4-element SubArray{String,1,Array{String,1},Tuple{Array{Int64,1}},false}:
    "a"
    "b"
    "b"
    "a"

Note two things in the code above.

1. Both, ``X_bal`` and ``y_bal``, are of type ``SubArray``. As
   such, they are just views into the corresponding original
   variable ``X`` or ``y``. The motivation for this behaviour is
   to avoid data movement until :func:`getobs`  is called.

2. The order of the observations in the resulting data subsets
   is the same as in the original data containers. This is no
   accident. If that behaviour is undesired, you can pass
   ``shuffle = true`` to the function.

If the type of the data is not sufficient information to be able
to extract the targets, one can specify a
target-extraction-function ``fun``, that is to be applied to each
individual observation. The behaviour when specifying or omitting
``fun`` is equivalent to its behaviour for :func:`eachtarget`,
because that is the function that is used internally by
:func:`undersample`.

A good example for a type that requires the parameter ``fun`` is
a ``DataTable``. Consider the following toy data container where
we know that the column ``:y`` contains the targets (see
:ref:`datatable` to make the following code work).

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(6), x2 = rand(6), y = [:a,:b,:b,:b,:b,:a])
   6×3 DataTables.DataTable
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.226582  │ 0.0443222   │ :a │
   │ 2   │ 0.504629  │ 0.722906    │ :b │
   │ 3   │ 0.933372  │ 0.812814    │ :b │
   │ 4   │ 0.522172  │ 0.245457    │ :b │
   │ 5   │ 0.505208  │ 0.11202     │ :b │
   │ 6   │ 0.0997825 │ 0.000341996 │ :a │

   julia> undersample(row->row[1,:y], dt)
   4×3 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.226582  │ 0.0443222   │ :a │
   │ 2   │ 0.504629  │ 0.722906    │ :b │
   │ 3   │ 0.522172  │ 0.245457    │ :b │
   │ 4   │ 0.0997825 │ 0.000341996 │ :a │

Of course, under-sampling the larger classes has the consequence
of decreasing the total size of the training set. After all, this
approach effectively discards perfectly usable training examples
for the sake of having a balanced data set. Alternatively, one
can also achieve a balanced class distribution by over-sampling
the smaller classes instead. To that end, we provide the function
:func:`oversample`. While this function effectively increases the
apparent size of the given data container, it does use the same
exact observations multiple times.

.. function:: oversample([fun], data, [fraction], [shuffle], [obsdim])

   Generate a re-balanced version of `data` by repeatedly
   sampling existing observations in such a way that every class
   will have at least `fraction` times the number observations of
   the largest class. This way, all classes will have a minimum
   number of observations in the resulting data set relative to
   what largest class has in the given (i.e. original) `data`.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the label for that
        observation.

   :param data: The object representing a labeled data container.

   :param Real fraction: \
        Optional. Minimum number of observations (as a fraction
        relative to the largest class) that every class should
        have. Defaults to ``1``, which implies completely
        balanced.

   :param bool shuffle: \
        Optional. Determines if the resulting data will be
        shuffled after its creation. If ``false``, then all the
        repeated samples will be together at the end, sorted by
        class. Defaults to ``true``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: An up-sampled version of `data` in the form of a lazy
            data subset. No data is copied until :func:`getobs`
            is called.

Let us again consider the toy data set from before, which
consists of a feature matrix ``X`` and a corresponding target
vector ``y``. Both variables are data containers with 6
observations.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> y = ["a", "b", "b", "b", "b", "a"]
   6-element Array{String,1}:
    "a"
    "b"
    "b"
    "b"
    "b"
    "a"

We previously "balanced" this data set by down-sampling it. To
show you an alternative, we can also up-sample it by repeating
observations for the under-represented class ``a``. You may
notice that this time the order of the observations will *not* be
preserved; it will even be shuffled. If that behaviour is
undesired, you can specify ``shuffle = false``.

.. code-block:: jlcon

   julia> X_bal, y_bal = oversample((X, y));

   julia> X_bal
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.11202      0.812814  0.226582  0.11202      0.933372 0.0443222  0.226582  0.505208
    0.000341996  0.245457  0.504629  0.000341996  0.522172 0.722906   0.504629  0.0997825

   julia> y_bal
   8-element SubArray{String,1,Array{String,1},Tuple{Array{Int64,1}},false}:
    "a"
    "b"
    "a"
    "a"
    "b"
    "b"
    "a"
    "b"


Similar to :func:`undersample` it is also possible to specify a
target-extraction-function ``fun``, that is to be applied to each
observation individually. Consider the following toy
``DataTable``, which we also used as an example data container to
demonstrate :func:`undersample`. For this particular data table
we know that the column ``:y`` contains the targets (see
:ref:`datatable` to make the following code work).

.. code-block:: jlcon

   julia> dt = DataTable(x1 = rand(6), x2 = rand(6), y = [:a,:b,:b,:b,:b,:a])
   6×3 DataTables.DataTable
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.226582  │ 0.0443222   │ :a │
   │ 2   │ 0.504629  │ 0.722906    │ :b │
   │ 3   │ 0.933372  │ 0.812814    │ :b │
   │ 4   │ 0.522172  │ 0.245457    │ :b │
   │ 5   │ 0.505208  │ 0.11202     │ :b │
   │ 6   │ 0.0997825 │ 0.000341996 │ :a │

   julia> oversample(row->row[1,:y], dt)
   8×3 DataTables.SubDataTable{Array{Int64,1}}
   │ Row │ x1        │ x2          │ y  │
   ├─────┼───────────┼─────────────┼────┤
   │ 1   │ 0.226582  │ 0.0443222   │ :a │
   │ 2   │ 0.505208  │ 0.11202     │ :b │
   │ 3   │ 0.0997825 │ 0.000341996 │ :a │
   │ 4   │ 0.504629  │ 0.722906    │ :b │
   │ 5   │ 0.933372  │ 0.812814    │ :b │
   │ 6   │ 0.226582  │ 0.0443222   │ :a │
   │ 7   │ 0.0997825 │ 0.000341996 │ :a │
   │ 8   │ 0.522172  │ 0.245457    │ :b │

While primarily intended for data container types, such as
``DataTable``, it is also useful for discretizing continuous
regression targets. Let's say you have a regression problem,
where you know that you have a small but important cluster of
observations with a particularly low target value. Given that
this cluster is under-represented, it could very well cause a
model to neglect those observations in order to improve its
performance on the rest of the data.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> y = [0.1, 0.95, 0.8, 0.9, 1.1, 0.11];

As can be seen in the code above, most observations have a target
value around ``1``, while just a small group of observations have
a target value around ``0.1``. In such a situation, you could use
the parameter ``fun`` to categorize the targets in such a way,
that will cause the under-represented "category" to be up-sampled
(or down-sampled) accordingly.

.. code-block:: jlcon

   julia> X_bal, y_bal = oversample(yi -> yi > 0.2, (X, y));

   julia> y_bal
   8-element SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}:
    0.11
    1.1
    0.1
    0.11
    0.95
    0.9
    0.1
    0.8

While the above example is a bit arbitrary, it highlights the
possibility that the functions :func:`undersample` and
:func:`oversample` can also be used to re-sample data container
with continuous targets.

A more common scenario would be when working with targets in the
form of a ``Matrix``. For instance, consider the following toy
data set where the targets ``Y`` are now a one-of-k encoded
matrix. In such a case we would like the be able to re-sample
without first having to convert ``Y`` to a different
class-encoding.

.. code-block:: jlcon

   julia> X = rand(2, 6)
   2×6 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814  0.11202
    0.504629  0.522172  0.0997825  0.722906   0.245457  0.000341996

   julia> Y = [1. 0. 0. 0. 0. 1.; 0. 1. 1. 1. 1. 0.]
   2×6 Array{Float64,2}:
    1.0  0.0  0.0  0.0  0.0  1.0
    0.0  1.0  1.0  1.0  1.0  0.0

Here we could use the function ``indmax`` to discretize the
individual target vectors on the fly. Remember that the
target-extraction-function is applied on each individual
observation in ``Y``. Since ``Y`` is a matrix, each observation
is a vector slice.

.. code-block:: jlcon

   julia> X_bal, Y_bal = oversample(indmax, (X, Y));

   julia> X_bal
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    0.226582  0.11202      0.11202      0.505208   0.226582  0.0443222  0.933372  0.812814
    0.504629  0.000341996  0.000341996  0.0997825  0.504629  0.722906   0.522172  0.245457

   julia> Y_bal
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
    1.0  1.0  1.0  0.0  1.0  0.0  0.0  0.0
    0.0  0.0  0.0  1.0  0.0  1.0  1.0  1.0
