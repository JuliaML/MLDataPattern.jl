.. _dataiterators:

Data Iterators
================

We hinted a few times before that we differentiate between two
kinds of data sources, namely *data containers* and *data
iterators*. We also briefly mentioned that a data source can
either be one, both, or neither of the two. So far, though, we
solely focused on data containers, and a special kind of data
container / data iterator hybrid that we called *data views*. If
we free ourselves from the notion that a data source has to know
how many observations it can provide, or that it has to
understand the concept of "accessing a specific observation", it
opens up a lot of new functionality that would otherwise be
infeasible.

In this document we will finally introduce those types of data
iterators, that do not make any other guarantees than what the
name implies: **iteration**. As such, they may not know how many
observation they can provide, or even understand what an
observation-index should be. One could ask what we could possibly
gain with such a type over the already introduced - and seemingly
more knowledgeable - data views. The answer is: **They address
different problems**. A very common and illustrative task, that
these data iterators are uniquely suited for, is continuous
random sampling from a data container.

Randomly sample Observations
-----------------------------

We previously introduced a type called :class:`ObsView`, which we
showed can be used to convert any data container to a data
iterator. As such, it makes it possible to, well, iterate over
the data container one observation at a time. Additionally, an
:class:`ObsView` also behaves like a vector, in that it allows to
use ``getindex`` to query a specific observation. By combining
:class:`ObsView` with :func:`shuffleobs`, we were also able to
iterate over all the observations from a data container in a
random order.

A different approach to iterating over a data container one
observation after another, is to continuously sample a single
observation from it. Per definition that means that the process
of determining the "next" observation is random. Thus, indexing a
specific observation of that iterator is ill defined. Therefore
data iterators in general only guarantee that they can be used as
a Julia iterator; every additional functionality is optional. One
such type data iterator that this package provides is called
:class:`RandomObs`.

.. class:: RandomObs <: ObsIterator

   A decorator type that transforms a data containers into a data
   iterator. Each iteration produces a randomly sampled
   observation from the given data container (with replacement).

   Note that each iteration returns the result of a
   :func:`datasubset`, which means that any data movement is
   delayed until :func:`getobs` is called.

.. function:: RandomObs(data, [count], [obsdim]) -> RandomObs

   Create an iterator that generates `count` randomly sampled
   observations from the given `data` container. In the case
   `count` is not provided, it will generate random samples
   indefinitely.

   :param data: The object representing a data container.

   :param Integer count: \
        Optional. The number of randomly sampled observations
        that the iterator will generate before stopping. If
        omitted, the iterator will generate randomly sampled
        batches forever.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Consider the following toy data vector ``x`` that has 5
observations. We will use simple values to make it easy to
see where each observation ends up.

.. code-block:: jlcon

   julia> x = collect(1.0:5)
   5-element Array{Float64,1}:
    1.0
    2.0
    3.0
    4.0
    5.0

Because ``x`` is a ``Vector`` it is considered a data container.
Thus we can pass it to :func:`RandomObs`. If we specify a
``count`` (i.e. limit the number of samples to generate), we can
use ``collect`` on it.

.. code-block:: jlcon

   julia> iter = RandomObs(x, count = 10)
   RandomObs(::Array{Float64,1}, 10, ObsDim.Last())
    Iterator providing 10 observations

   julia> xnew = collect(iter)
   10-element Array{SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false},1}:
    4.0
    4.0
    1.0
    5.0
    2.0
    5.0
    1.0
    2.0
    1.0
    5.0

   julia> xnew[1]
   0-dimensional SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}:
    4.0

Notice two things in the code above.

- The observations from ``x`` are sampled randomly with
  replacement. That means the same observation can occur in
  ``xnew`` once, multiple times, or not at all.

- Each sampled observation is actually a lazy subset (i.e. a
  ``SubArray``) into the original data container ``x``. To get
  the underlying data you need to use :func:`getobs` on the
  result.

The constructor parameter ``count`` is optional and can be
omitted. If that is the case, then the resulting iterator will
continue to sample random observations forever, or until
interrupted.

.. code-block:: jlcon

   julia> iter = RandomObs(x)
   RandomObs(::Array{Float64,1}, ObsDim.Last())
    Iterator providing Inf observations

   julia> collect(iter) # can't collect infinite iterator
   ERROR: MethodError: no method matching _collect(::UnitRange{Int64}, ::MLDataPattern.RandomObs{SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false},Array{Float64,1},LearnBase.ObsDim.Last,Base.IsInfinite}, ::Base.HasEltype, ::Base.IsInfinite)

   julia> collect(take(iter, 5))
   5-element Array{SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false},1}:
    4.0
    4.0
    1.0
    5.0
    2.0

Similar to an :class:`ObsView`, it is also possible to use a
``Tuple`` to group data containers together on a per-observation
level. This will cause each iteration to return a ``Tuple`` of
equal length and ordering.

.. code-block:: jlcon

   julia> y = [:a, :b, :c, :d, :e];

   julia> iter = RandomObs((x, y), count = 5)
   RandomObs(::Tuple{Array{Float64,1},Array{Symbol,1}}, 5, (ObsDim.Last(),ObsDim.Last()))
    Iterator providing 5 observations

   julia> collect(iter)
   5-element Array{Tuple{SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false},SubArray{Symbol,0,Array{Symbol,1},Tuple{Int64},false}},1}:
    (4.0,:d)
    (4.0,:d)
    (1.0,:a)
    (5.0,:e)
    (2.0,:b)

In case of skewed class distributions we offer an alternative
iterator called :class:`BalancedObs`, which samples from each
label uniformly.

.. code-block:: jlcon

   julia> y = [:a, :a, :a, :a, :a, :a, :a, :a, :b, :b];

   julia> iter = BalancedObs((1:10, y), count = 6)
   BalancedObs(::Tuple{UnitRange{Int64},Array{Symbol,1}}, 6, (ObsDim.Last(), ObsDim.Last()))
    Iterator providing 6 observations

   julia> collect(iter)
   6-element Array{Tuple{SubArray{Int64,0,UnitRange{Int64},Tuple{Int64},false},SubArray{Symbol,0,Array{Symbol,1},Tuple{Int64},false}},1}:
    (10, :b)
    (4, :a)
    (9, :b)
    (7, :a)
    (8, :a)
    (9, :b)

Randomly sample Mini-Batches
------------------------------

Similarly to :class:`BatchView`, an object of type
:class:`RandomBatches` can be used as an iterator that produces a
mini-batch of fixed size in each iteration. In contrast to
:class:`BatchView`, however, :class:`RandomBatches` generates
completely random mini-batches, in which the containing
observations are generally not adjacent to each other in the
original dataset.

.. class:: RandomBatches <: BatchIterator

   A decorator type that transforms a data container into a data
   iterator, that on each iteration returns a batch of fixed size
   containing randomly sampled observation from the given data
   container (with replacement).

   Each iteration returns the result of calling
   :func:`datasubset`, which means that any data movement is
   delayed until :func:`getobs` is called.

.. function:: RandomBatches(data, [size], [count], [obsdim]) -> RandomBatches

   Create an iterator that generates `count` randomly sampled
   batches from the given `data` container using a batch-size of
   `size`. In the case `count` is not provided, it will generate
   random batches indefinitely.

   :param data: The object representing a data container.

   :param Integer size: \
        Optional. The batch-size of each batch. I.e. the number
        of randomly sampled observations in each batch.

   :param Integer count: \
        Optional. The number of randomly sampled batches that the
        iterator will generate before stopping. If omitted, the
        iterator will generate randomly sampled observations
        forever.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Consider our simple toy data vector ``x`` again, that we used
before to motivate :class:`RandomObs`.

.. code-block:: jlcon

   julia> x = collect(1.0:5)
   5-element Array{Float64,1}:
    1.0
    2.0
    3.0
    4.0
    5.0

Because ``x`` is considered a data container, it can be used to
produce random batches with :class:`RandomBatches`. We can use
the parameter ``size`` to specify how many observations each
mini-batch should contain. If we also specify a ``count`` (i.e.
limit the number of mini-batches to generate), we can use
``collect`` on the result.

.. code-block:: jlcon

   julia> iter = RandomBatches(x, size = 3, count = 10)
   RandomBatches(::Array{Float64,1}, 3, 10, ObsDim.Last())
    Iterator providing 10 batches of size 3

   julia> collect(iter)
   10-element Array{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},1}:
    [4.0,4.0,1.0]
    [5.0,2.0,5.0]
    [1.0,2.0,1.0]
    [5.0,2.0,4.0]
    [1.0,1.0,2.0]
    [2.0,5.0,2.0]
    [3.0,2.0,1.0]
    [2.0,5.0,4.0]
    [1.0,2.0,4.0]
    [5.0,5.0,2.0]

The constructor parameter ``count`` is optional and can be
omitted. If that is the case, then the resulting iterator will
continue to sample random mini-batches forever, or until
interrupted.

.. code-block:: jlcon

   julia> iter = RandomBatches(x, size = 3)
   RandomBatches(::Array{Float64,1}, 3, ObsDim.Last())
    Iterator providing Inf batches of size 3

   julia> collect(iter) # can't collect infinite iterator
   ERROR: MethodError: no method matching _collect(::UnitRange{Int64}, ::MLDataPattern.RandomBatches{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},Array{Float64,1},LearnBase.ObsDim.Last,Base.IsInfinite}, ::Base.HasEltype, ::Base.IsInfinite)

   julia> collect(take(iter, 5))
   5-element Array{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},1}:
    [4.0,4.0,1.0]
    [5.0,2.0,5.0]
    [1.0,2.0,1.0]
    [5.0,2.0,4.0]
    [1.0,1.0,2.0]

Because the utilized data container ``x`` is a vector, each
mini-batch is a one-dimensional ``SubArray`` (i.e. a lazy subset
into ``x``). The type of each mini-batch depends on the given
data container. For example if we instead use a feature *matrix*
``X``, each mini-batch would be a two-dimensional ``SubArray``.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> iter = RandomBatches(X, size = 3, count = 10)
   RandomBatches(::Array{Float64,2}, 3, 10, ObsDim.Last())
    Iterator providing 10 batches of size 3

   julia> collect(iter)
   10-element Array{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},1}:
    [0.226582 0.933372 0.933372; 0.504629 0.522172 0.522172]
    [0.812814 0.933372 0.505208; 0.245457 0.522172 0.0997825]
    [0.933372 0.226582 0.933372; 0.522172 0.504629 0.522172]
    [0.812814 0.0443222 0.226582; 0.245457 0.722906 0.504629]
    [0.933372 0.0443222 0.812814; 0.522172 0.722906 0.245457]
    [0.812814 0.933372 0.0443222; 0.245457 0.522172 0.722906]
    [0.226582 0.933372 0.226582; 0.504629 0.522172 0.504629]
    [0.0443222 0.812814 0.505208; 0.722906 0.245457 0.0997825]
    [0.226582 0.812814 0.812814; 0.504629 0.245457 0.245457]
    [0.812814 0.812814 0.0443222; 0.245457 0.245457 0.722906]

It is also possible to link multiple different data containers
together on an per-observation level. This way they can be
sampled from as one coherent unit. To do that, simply put all the
relevant data container into a single ``Tuple``, before passing
it to :func:`RandomBatches`. This will cause each iteration to
return a ``Tuple`` of equal length and ordering.

.. code-block:: jlcon

   julia> y = [:a, :b, :c, :d, :e];

   julia> iter = RandomBatches((x, y), size = 3, count = 5)
   RandomBatches(::Tuple{Array{Float64,1},Array{Symbol,1}}, 3, 5, (ObsDim.Last(),ObsDim.Last()))
    Iterator providing 5 batches of size 3

   julia> collect(iter)
   5-element Array{Tuple{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},SubArray{Symbol,1,Array{Symbol,1},Tuple{Array{Int64,1}},false}},1}:
    ([4.0,4.0,1.0],Symbol[:d,:d,:a])
    ([5.0,2.0,5.0],Symbol[:e,:b,:e])
    ([1.0,2.0,1.0],Symbol[:a,:b,:a])
    ([5.0,2.0,4.0],Symbol[:e,:b,:d])
    ([1.0,1.0,2.0],Symbol[:a,:a,:b])

The fact that the observations within each mini-batch are
randomly sampled has an important consequences. Because
observations are sampled with replacement, it is likely that some
observation(s) occur multiple times within the same mini-batch.
This may or may not be an issue, depending on the use-case. In
the presence of online data-augmentation strategies, this fact
should usually not have any noticeable impact.

The BufferGetObs Type
------------------------

You may have noticed that all the data iterators and data views,
:class:`RandomObs`, :class:`RandomBatches`, :class:`ObsView`, and
:class:`BatchView`, return a lazy data subset for every
iteration. This is useful in general, because it avoids data
access and memory allocation until the user makes a conscious
decision to do so by calling :func:`getobs`. That said, in many
use cases it would be convenient if we could tell a data iterator
(or data view) to return the actual data in each iteration,
instead of a lazy subset. To that end, this package provides a
special iterator decorator that is itself an iterator (just
"iterator"; it is not a "data iterator") called
:class:`BufferGetObs`.

.. class:: BufferGetObs

   A stateful iterator that decorates an inner ``iterator``. When
   iterated over the type stores the output of
   ``next(iterator,state)`` into a ``buffer`` using
   ``getobs!(buffer, ...)``. Depending on the type of data
   provided by ``iterator`` this may be more memory efficient
   than ``getobs(...)``. In the case of array data, for example,
   this allows for cache-efficient processing of each element
   without allocating a temporary array.

   Note that not all types of data support buffering, because it
   is the developers's choice to opt-in and implement a custom
   :func:`getobs!`. For those types that do not provide a custom
   :func:`getobs!`, the ``buffer`` will be ignored and the result
   of ``getobs(...)`` returned.

.. function:: BufferGetObs(iterator, [buffer]) -> BufferGetObs

   :param iterator: Some type that implements the iterator
        pattern, and for which every generated element supports
        :func:`getobs`

   :param buffer: Optional. If the elements of `iterator` support
        :func:`getobs!`, then this buffer is used as temporary
        storage on every iteration. Defaults to the result of
        :func:`getobs` on the first element of `iterator`.

Let us take a look at an example where :class:`BufferGetObs`
shines. Consider the following toy feature matrix ``X`` that
contains 5 observation with 3 features each. Notice how in this
example each row denotes a single observation.

.. code-block:: jlcon

   julia> X = rand(5, 3)
   5×3 Array{Float64,2}:
    0.226582  0.0997825  0.11202
    0.504629  0.0443222  0.000341996
    0.933372  0.722906   0.380001
    0.522172  0.812814   0.505277
    0.505208  0.245457   0.841177

Given that arrays in Julia are in column-major order, the
features of each observations are not a continuous block of
memory. This fact by itself need not be an issue. For example, if
we would want to iterate over the data container one observation
at a time, we could still use :func:`obsview` without noticing
any obvious differences.

.. code-block:: jlcon

   julia> ov = obsview(X, obsdim = 1)
   5-element obsview(::Array{Float64,2}, ObsDim.Constant{1}()) with element type SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    [0.226582,0.0997825,0.11202]
    [0.504629,0.0443222,0.000341996]
    [0.933372,0.722906,0.380001]
    [0.522172,0.812814,0.505277]
    [0.505208,0.245457,0.841177]

   julia> ov[2] # access second observation
   3-element SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    0.504629
    0.0443222
    0.000341996

On the other hand, if need to interact with some C library, which
requires us to pass to it a proper continuous array, then we
can't just use this ``SubArray`` as it is. Luckily, we could just
use :func:`getobs` on each subset and pass the resulting
``Array`` to the C library.

.. code-block:: julia

   for xv in obsview(X, obsdim = 1)
       x = getobs(xv)
       # pass x to some c library
   end

The remaining annoyance with the above code is that it allocates
temporary memory on each iteration. In a performance critical
inner loop this is undesired and could have a significant
influence on the performance. To avoid that problem, we can
preallocate a buffer and reuse it in every iteration with
:func:`getobs!`.

.. code-block:: julia

   x = Vector{Float64}(3)
   for xv in obsview(X, obsdim = 1)
       getobs!(x, xv)
       # pass x to some c library
   end

This should give us pretty good performance. This pattern is so
common, however, that this package provides a convenience
implementation for it, namely :class:`BufferGetObs`.

.. code-block:: julia

   for x in BufferGetObs(obsview(X, obsdim = 1), Vector{Float64}(3))
       # pass x to some c library
   end

The nice thing about using :class:`BufferGetObs` is that it
doesn't even require us to manually provide a preallocated
buffer. If omitted, :class:`BufferGetObs` simply reuses the
result of :func:`getobs` from the first element.

.. code-block:: julia

   for x in BufferGetObs(obsview(X, obsdim = 1))
       # pass x to some c library
   end

Furthermore, because it is so common to use :class:`BufferGetObs`
in combination with either :class:`ObsView` or
:class:`BatchView`, we provide convenience functions for both.
More concretely, the functions :func:`eachobs` and
:func:`eachbatch` simply translate to
``BufferGetObs(ObsView(...))`` and
``BufferGetObs(BatchView(...))`` respectively.

.. function:: eachobs(data, [obsdim]) -> BufferGetObs

   Iterate over `data` one observation at a time using
   :class:`ObsView`. In contrast to :class:`ObsView`, each
   iteration returns the result of :func:`getobs` (i.e. actual
   data). If supported by the type of `data`, a buffer will be
   preallocated and reused every iteration for memory efficiency.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: The result of ``BufferGetObs(ObsView(data, obsdim))``

.. function:: eachbatch(data, [size], [count], [obsdim]) -> BufferGetObs

   Iterate over `data` one batch at a time using
   :class:`BatchView`. In contrast to :class:`BatchView`, each
   iteration returns the result of :func:`getobs` (i.e. actual
   data). If supported by the type of data, a buffer will be
   preallocated and reused for memory efficiency.

   The (constant) batch-size can be either provided directly
   using `size` or indirectly using `count`, which derives the
   size based on :func:`nobs`. In the case that the size of the
   `data` is not dividable by the specified (or inferred) `size`,
   the remaining observations will be ignored.

   :param data: The object representing a data container.

   :param Integer size: Optional. The number of observations in
                        each batch.

   :param Integer count: \
        Optional. The number of batches that should be used. This
        will also we the length of the return value.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

   :return: The result of ``BufferGetObs(BatchView(data, size, count, obsdim))``
