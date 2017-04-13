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


A different approach to iterating over a data container, would be
to iterate over random

.. class:: RandomObs <: ObsIterator

   A decorator type that transforms a data containers into a data
   iterator, that returns a randomly sampled observation from the
   given data container (with replacement).

   Each iteration returns the result of a :func:`datasubset`,
   which means that any data movement is delayed until
   :func:`getobs` is called.

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

Randomly sample Mini-Batches
------------------------------

The purpose of :class:`RandomBatches` is to provide a generic
:class:`DataIterator` specification for labeled and unlabeled
randomly sampled mini-batches that can be used as an iterator.
In contrast to :class:`BatchView`, :class:`RandomBatches`
generates completely random mini-batches, in which the containing
observations are generally not adjacent to each other in the
original dataset.

The fact that the observations within each mini-batch are
uniformly sampled has an important consequences. Because
observations are independently sampled, it is likely that some
observation(s) occur multiple times within the same mini-batch.
This may or may not be an issue, depending on the use-case. In
the presence of online data-augmentation strategies, this fact
should usually not have any noticible impact.

.. class:: RandomBatches <: BatchIterator

   A decorator type that transforms a data containers into a data
   iterator, that on each iteration returns a batch of fixed size
   containing randomly sampled observation from the given data
   container (with replacement).

   Each iteration returns the result of a :func:`datasubset`,
   which means that any data movement is delayed until
   :func:`getobs` is called.

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

The BufferGetObs Type
------------------------

.. class:: BufferGetObs

   A stateful iterator that decorades an inner ``iterator``. When
   iterated over the type stores the output of
   ``next(iterator,state)`` into a ``buffer`` using
   ``getobs!(buffer, ...)``. Depending on the type of data
   provided by ``iterator`` this may be more memory efficient
   than ``getobs(...)``. In the case of array data, for example,
   this allows for cache-efficient processing of each element
   without allocating a temporary array.

   Note that not all types of data support buffering, because it
   is the author's choice to opt-in and implement a custom
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
