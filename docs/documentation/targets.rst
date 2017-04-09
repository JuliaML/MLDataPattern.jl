Labeled Data Container
=======================

Depending on the domain specific problem and the data one is
working with, a data container may or may not contain
**targets**. A target (singular) is a piece of information about
a single observation that represents the desired output (or
correct answer) for that specific observation. So if targets are
involved, then we find ourselves in the realm of supervised
learning. This includes both, classification (predicting
categorical output) and regression (predicting real-valued
output).

Dealing with targets in a generic way was quite a design
challenge for us. There are a lot of aspects to consider and
issues to address in order to achieve an extensible and flexible
package architecture.

- Some data container may not contain any targets or even
  understand what targets are.

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
  container that represents a directory of images in the file
  system, in which each sub-directory contains the images of a
  single class. In that scenario, the targets are known from the
  directory names (i.e. the metadata). As such it would be far
  more efficient if the data container can make use of this
  information, instead of having to load an actual image from the
  disk just to access its target. Remember that targets are not
  only needed during training itself, but also for data
  partitioning and resampling.

Query Target(s)
-------------------

The targets logic is in some ways a bit more complex than the
:func:`getobs` logic. The main reason for this is that we want to
support a wide variety of data container types and data
scenarios. To that end we provide the function :func:`targets`.
Note that this function serves as a porcelain interface and
should not be extended directly.

.. function:: targets(data, [obsdim])

   Extract the concrete targets from `data` and return them.

   This function is eager in the sense that it will always call
   :func:`getobs` unless a custom method for ``gettargets`` (see
   later) is implemented for the type of `data`. This will make
   sure that actual values are returned (in contrast to
   placeholders such as :class:`DataSubset` or ``SubArray``).

   In other words, the returned values must be in the form
   intended to be passed as-is to some resampling strategy or
   learning algorithm.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

If ``data`` is a tuple, then the convention is that the last
element of the tuple contains the targets and the function is
recursed once (and only once).

.. code-block:: jlcon

   julia> targets(([1,2], [3,4]))
   2-element Array{Int64,1}:
    3
    4

   julia> targets(([1,2], ([3,4], [5,6])))
   ([3,4],[5,6])

If the type alone is not sufficient information to be able to
return the targets, one must specify a target-extraction-function
``fun`` that is to be applied to each observations. This function
must be passed as the first parameter to :func:`targets`.

.. function:: targets(fun, data, [obsdim]) -> Vector

   Extract the concrete targets from the observations in `data`
   by applying `fun` on each observation individually. The
   extracted targets are returned as a ``Vector``, which
   preserves the order of the observations from `data`.

   :param fun: \
        A callable object (usually a function) that should
        be applied to each observation individually in order to
        extract or compute the target for that observation.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

.. code-block:: jlcon

   julia> targets(indmax, [1 0 1; 0 1 0])
   3-element Array{Int64,1}:
    1
    2
    1

Note that if the optional first parameter is passed to
:func:`targets`, it will always be applied to the observations,
and not the container. In other words, the first parameter is
applied to each observation individually and not to the data as a
whole. In general this means that the return type changes
drastically even if passing a no-op function.

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

The optional parameter ``obsdim`` can be used to specify which
dimension denotes the observations, if that concept makes sense
for the type of ``data``.

.. code-block:: jlcon

   julia> targets(indmax, [1 0; 0 1; 1 0], obsdim=1)
   3-element Array{Int64,1}:
    1
    2
    1

   julia> targets(indmax, [1 0; 0 1; 1 0], ObsDim.First())
   3-element Array{Int64,1}:
    1
    2
    1

Note how ``obsdim`` can either be provided using type-stable
positional arguments from the namespace ``ObsDim``, or by using a
more flexible and convenient keyword argument. We will discuss
observation dimensions in more detail in a later section.

Iterate over Targets
---------------------

In some situations one only wants to iterate over the targets
instead of computing all of them at once. In those situations it
would be beneficial to avoid allocation temporary memory. To that
end we provide the function :func:`eachtarget`, which returns a
``Base.Generator``, that when iterated over returns each target
in ``data`` once and in the correct order.

.. function:: eachtarget([fun], data, [obsdim]) -> Generator

   Return a ``Base.Generator`` that iterates over all targets in
   `data` once and in the right order. If `fun` is provided it
   will be applied to each observation in data.

   :param fun: \
        Optional. A callable object (usually a function) that
        should be applied to each observation individually in
        order to extract or compute the target for that
        observation.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

.. code-block:: jlcon

   julia> iter = eachtarget(([1,2], [3,4]))
   Base.Generator{UnitRange{Int64},MLDataUtils.##79#80{2,Tuple{Array{Int64,1},Array{Int64,1}},Tuple{LearnBase.ObsDim.Last,LearnBase.ObsDim.Last}}}(MLDataUtils.#79,1:2)

   julia> collect(iter)
   2-element Array{Int64,1}:
    3
    4

   julia> iter = eachtarget([1 0; 0 1; 1 0])
   Base.Generator{UnitRange{Int64},MLDataUtils.##75#76{Array{Int64,2},LearnBase.ObsDim.Last}}(MLDataUtils.#75,1:2)

   julia> collect(iter)
   2-element Array{Array{Int64,1},1}:
    [1,0,1]
    [0,1,0]

   julia> iter = eachtarget(indmax, [1 0; 0 1; 1 0])
   Base.Generator{MLDataUtils.ObsView{SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true},Array{Int64,2},LearnBase.ObsDim.Last},MLDataUtils.##83#84{Base.#indmax}}(MLDataUtils.#83,SubArray{Int64,1,Array{Int64,2},Tuple{Colon,Int64},true}[[1,0,1],[0,1,0]])

   julia> collect(iter)
   2-element Array{Int64,1}:
    1
    2

Just like for :func:`target`, the optional parameter ``obsdim``
can be used to specify which dimension denotes the observations,
if that concept makes sense for the type of ``data``.

.. code-block:: jlcon

   julia> iter = eachtarget(indmax, [1 0; 0 1; 1 0], obsdim=1)
   Base.Generator{MLDataUtils.ObsView{SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Colon},true},Array{Int64,2},LearnBase.ObsDim.Constant{1}},MLDataUtils.##83#84{Base.#indmax}}(MLDataUtils.#83,SubArray{Int64,1,Array{Int64,2},Tuple{Int64,Colon},true}[[1,0],[0,1],[1,0]])

   julia> collect(iter)
   3-element Array{Int64,1}:
    1
    2
    1

Support for Custom Types
--------------------------

A package author has two ways to customize the logic behind
:func:`targets` for their own data types:

1. Implement ``gettargets`` for the data container type,
   which bypasses :func:`getobs` entirely.

2. Implement ``gettarget`` for the observation type,
   which is applied on the result of :func:`getobs`.

Here are two example scenarios that benefit from custom methods.
The first one for ``gettargets``, and the second one for
``gettarget``. Note again that these functions are internal and
only intended to be *extended* by the user (and **not** called).
A user should not use them directly but instead always call
:func:`targets`.

See the corresponding doc-strings for more information.

Use-Case 1: Custom Directory Based Image Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say you write a custom data container that describes a
directory on your hard-drive. Each subdirectory contains a set of
large images that belong to a single class (the directory name).
This kind of data container only loads the images itself if they
are actually needed (so on :func:`getobs`). The targets however
are part of the metadata that is always loaded. So if we are only
interested in the targets (for example for data partitioning or
resampling), then we would like to avoid calling :func:`getobs`
if possible. We can do that by implementing a custom method for
``gettargets``.

.. code-block:: julia

   MLDataUtils.gettargets(::MyImageSource, i) = ...

This allows a user to do just that. In other words it allows to
provide the targets of some observation(s) without ever calling
:func:`getobs`.

Use-Case 2: Symbol Support for DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DataFrames are a kind of data container, where the targets are
as much part of the data as the features are (in contrast to
Use-Case 1). Here we are fine with :func:`getobs` being called.
However, we still need to say which column actually describes
the features. We can do this by passing a function
``targets(row->row[1,:Y], dataframe)``, or we can provide a
convenience syntax by overloading ``gettarget``.

.. code-block:: julia

   MLDataUtils.gettarget(col::Symbol, df::DataFrame) = df[1,col]

This now allows us to call ``targets(:Y, dataframe)``. While not
strictly necessary in this case, it can be quite useful for
special types of observations, such as ``ImageMeta``.

