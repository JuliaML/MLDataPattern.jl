.. _folds:

Repartitioning Strategies
================================

Most non-trivial machine learning experiments require some form
of model tweaking *prior* to training. A particularly common
scenario is when the model (or algorithm) has hyper parameters
that need to be specified manually. The process of searching for
suitable hyper parameters is a sub-task of what we call *model
selection*.

If model selection is part of the experiment, then it is quite
likely that a simple train/test split will not be effective
enough to achieve results that are representative for new, unseen
data. The reason for this is subtle, but very important. If the
hyper parameters are chosen based on how well the corresponding
model performs on the test set, then information about the test
set is actively fed back into the model. This is because the test
set is used several times *and* decisions are made based on what
was observed. In other words: the test set participates in an
aspect of the training process, namely the model selection.
Consequently, the results on the test set become less
representative for the expected results on new, unseen data. To
avoid causing this kind of manual overfitting, one should instead
somehow make use of the training set for such a model selection
process, while leaving the test set out of it completely.
Luckily, this can be done quite effectively using by a
repartitioning strategy, such as a :math:`k`-folds, to perform
cross-validation.

We will start by discussing the terminology that is used
throughout this document. More importantly, we will define how
the various terms are interpreted in the context of this package.
The rest of this document will then focus on how these concepts
are implemented and exposed to the user. There we will start by
introducing some low-level helper methods for computing the
required subset-assignment indices. We will then use those
"assignments" to motivate a type called :class:`FoldsView`, which
can be configured to represent almost any kind of repartitioning
strategy for a given data container. After discussing those
basics, we will introduce the high-level methods that serve as a
convenience layer around :class:`FoldsView` and the low-level
functionality.

Terms and Definitions
--------------------------

Before we dive into the provided functionality, let us quickly
discuss some terminology. A few of the involved terms are often
used quite casually in conversations, and thus easy to mix up. In
general that doesn't cause much confusion, but since parts of
this document are concerned with low-level functionality, we deem
it important that we share the same wording.

- When we have multiple disjoint subsets of the same data
  container (or tuple of data containers), we call the grouping
  of those subsets a **partition**. That is, a partition is a
  particular outcome of assigning the observations from some data
  container to multiple disjoined subsets. In contrast to the
  formal definition in mathematics, we do allow the same
  observation to occur multiple times in the *same* subset.

  For instance the function :func:`splitobs` creates a single
  partition in the form of a tuple. More concretely, the
  following code snippet creates a partition with two subsets
  from a given toy data-vector that has 5 observations.

  .. code-block:: jlcon

     julia> partition = splitobs([1,2,3,4,5], at = 0.6)
     ([1,2,3],[4,5])

- In the context of this package, a **repartitioning strategy**
  describes a particular "system" for reassigning the
  observations of a data container (or tuple of data containers)
  to a training subset and a validation subset *multiple times*.
  So in contrast to a simple train/validation split, the data
  isn't just partitioned once, but in multiple different
  configurations. In other words, the result of a repartitioning
  strategy are multiple different partitions of the same data. We
  use the term "repartitioning strategy" instead of "resampling
  strategy" to emphasize that the subsets of each partition are
  disjoint.

  An example for performing a really simply repartitioning
  strategy would be to create a sequences of random
  train/validation partitions of some given data. The following
  code snippet computes 3 partitions (which are also often
  referred to as *folds*) for such a strategy on a random toy
  data-vector ``y`` that has 5 observations in it.

  .. code-block:: jlcon

     julia> y = rand(5);

     julia> folds = [splitobs(shuffleobs(y), at = 0.6) for i in 1:3]
     3-element Array{Tuple{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}},1}:
      ([0.933372,0.522172,0.505208],[0.504629,0.226582])
      ([0.226582,0.504629,0.505208],[0.522172,0.933372])
      ([0.505208,0.504629,0.933372],[0.226582,0.522172])

- The result of a repartitioning strategy can be described
  through a sequences of *subset assignment indices*, or short
  **assignments**. An assignment (singular) describes a partition
  that is valid for any data container of size :math:`N` by using
  indices from the set :math:`\{1,2,...,N\}`. For instance, if a
  single partition should consist of two subsets, then the
  corresponding assignment (singular), is made up of two vectors
  of indices, each vector describing the content of one subset.
  Because of this, it is also fair to think about the result of a
  repartitioning strategy as two sequences, one for the *training
  assignments* and a corresponding sequence for the *validation
  assignments*.

  To give a concrete example of such assignment sequences,
  consider the result of calling ``kfolds(6, 3)`` (see code
  below). It will compute the training assignments ``train_idx``
  and the corresponding validation assignments ``val_idx`` for a
  3-fold repartitioning strategy that is applicable to any data
  container that has 6 observations in it.

  .. code-block:: jlcon

     julia> train_idx, val_idx = kfolds(6, 3)
     ([[3,4,5,6],[1,2,5,6],[1,2,3,4]], [[1,2],[3,4],[5,6]])

     julia> train_idx # sequence of training assignments
     3-element Array{Array{Int64,1},1}:
      [3,4,5,6]
      [1,2,5,6]
      [1,2,3,4]

     julia> val_idx # sequence of validation assignments
     3-element Array{Array{Int64,1},1}:
      [1,2]
      [3,4]
      [5,6]

- The result of applying a sequence of assignments to some data
  container (or tuple of data containers) is a sequence of
  **folds**. In the context of this package the term "fold" is
  almost interchangeable with "partition". In contrast to a
  partition, however, the term "fold" implies that there exist
  more than one.

  For instance, let us consider manually applying the assignments
  (which we have computed above) to some random toy data-vector
  ``y`` of appropriate length 6.

  .. code-block:: jlcon

     julia> y = rand(6)
     6-element Array{Float64,1}:
      0.226582
      0.504629
      0.933372
      0.522172
      0.505208
      0.0997825

     julia> folds = map((t,v)->(view(y,t),view(y,v)), train_idx, val_idx)
     3-element Array{Tuple{SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false},SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}},1}:
      ([0.933372,0.522172,0.505208,0.0997825],[0.226582,0.504629])
      ([0.226582,0.504629,0.505208,0.0997825],[0.933372,0.522172])
      ([0.226582,0.504629,0.933372,0.522172],[0.505208,0.0997825])

Naturally, the above code snippets just serve as examples to
motivate the problem. This package implements a number of
functions that provide the necessary functionality in a more
intuitive and convenient manner.

Computing K-Folds Indices
--------------------------

A particularly popular validation scheme for model selection is
*k-fold cross-validation*; the first step of which is dividing
the data set into :math:`k` roughly equal-sized parts. Each model
is fit :math:`k` times, while each time a different part is left
out during training. The left out part instead serves as a
validation set, which is used to compute the metric of interest.
The validation results of the :math:`k` trained model-instances
are then averaged over all :math:`k` folds and reported as the
performance for the particular set of hyper parameters.

Before we go into details about the partitioning or, later, the
validation aspects, let us first consider how to compute the
underlying representation. In particular how to compute the
**assignments** that can then be used to create the folds. For
that purpose we provide a helper method for the function
:func:`kfolds`.

.. function:: kfolds(n, [k = 5]) -> Tuple

   Compute the train/validation assignments for `k` partitions of
   `n` observations, and return them in the form of two vectors.
   The first vector contains the sequence of training assignments
   (i.e. the indices for the training subsets), and the second
   vector the sequence of validation assignments (i.e. the
   indices for the validation subsets).

   Each observation is assigned to a validation subset once (and
   only once). Thus, a union over all validation assignments
   reproduces the full range ``1:n``. Note that there is no
   random placement of observations into subsets, which means
   that adjacent observations are likely part of the same subset.

   *Note*: The sizes of the validation subsets may differ by up
   to 1 observation depending on if the total number of
   observations `n` is dividable by `k`.

   :param Integer n: Total number of observations to compute the
                     folds for.

   :param Integer k: Optional. The number of folds to compute. A
                     general rule of thumb is to use either ``k =
                     5`` or ``k = 10``. Must be within the range
                     ``2:n``. Defaults to ``k = 5``.

   :return: A ``Tuple`` of two ``Vector``. Both vectors are of
            length `k`, where each element is also a vector. The
            first vector represents the sequence of training
            assignments, and the second vector the sequence
            of validation assignments.

Invoking :func:`kfolds` with an integer as first parameter - as
outlined above - will compute the assignments for a
:math:`k`-folds repartitioning strategy. For instance, the
following code will compute the sequences of training- and
validation assignments for 10 observations and 4 folds.

.. code-block:: jlcon

   julia> train_idx, val_idx = kfolds(10, 4); # 10 observations, 4 folds

   julia> train_idx
   4-element Array{Array{Int64,1},1}:
    [4,5,6,7,8,9,10]
    [1,2,3,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> val_idx
   4-element Array{UnitRange{Int64},1}:
    1:3
    4:6
    7:8
    9:10

As we can see, there is no actual data set involved yet. We just
computed assignments that are applicable to *any* data set that
has exactly 10 observations in it. The important thing to note
here is that while the indices in ``train_idx`` overlap, the
indices in ``val_idx`` do not, and further, all 10
observation-indices are part of one (and only one) element of
``val_idx``.

Computing Leave-Out Indices
--------------------------------

A different way to think about a :math:`k`-folds repartitioning
strategy is in terms of the size of each validation subset.
Instead of specifying the number of folds directly, we specify
how many observations we would like to be in each validation
subset. While the resulting assignments are equivalent to the
result of some particular :math:`k`-folds scheme, it is sometimes
referred to as *leave-p-out partitioning*. A particularly common
version of which is leave-one-out, where we set the validation
subset size to 1 observation.

.. function:: leaveout(n, [size = 1]) -> Tuple

   Compute the train/validation assignments for ``k ≈ n/size``
   repartitions of `n` observations, and return them in the form
   of two vectors. The first vector contains the sequence of
   training assignments (i.e. the indices for the training
   subsets), and the second vector the sequence of validation
   assignments (i.e. the indices for the validation subsets).

   Each observation is assigned to the validation subset once
   (and only once). Furthermore, each validation subset will have
   either `size` or `size` + 1 observations assigned to it.

   Note that there is no random placement of observations into
   subsets, which means that adjacent observations are likely
   part of the same subset.

   :param Integer n: Total number of observations to compute the
                     folds for.

   :param Integer size: Optional. The desired number of
                        observations in each validation subset.
                        Defaults to ``size = 1``.

   :return: A ``Tuple`` of two ``Vector``. Both vectors are of
            queal length, where each element is also a vector.
            The first vector represents the sequence of training
            assignments, and the second vector the sequence of
            validation assignments.

Invoking :func:`leaveout` with an integer as first parameter will
compute the sequence of assignments for a :math:`k`-folds
repartitioning strategy. For example, the following code will
assign the indices of 10 observations to as many partitions as it
takes such that every validation subset contains approximately 2
observations.

.. code-block:: jlcon

   julia> train_idx, val_idx = leaveout(10, 2);

   julia> train_idx
   5-element Array{Array{Int64,1},1}:
    [3,4,5,6,7,8,9,10]
    [1,2,5,6,7,8,9,10]
    [1,2,3,4,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> val_idx
   5-element Array{UnitRange{Int64},1}:
    1:2
    3:4
    5:6
    7:8
    9:10

Just like before, there is no actual data set involved here. We
simply computed assignments that are applicable to *any* data set
that has exactly 10 observations in it. Note that for the above
example the result is equivalent to calling ``kfolds(10, 5)``.

.. _foldsview:

The FoldsView Type
-----------------------

So far we focused on just computing the sequence of assignments
for various repartition strategies, without any regard to an
actual data set. Instead, we just specified the total number of
observations. Naturally that is only one part of the puzzle. What
we really care about after all, is the repartitioning of an
actual data set. To that end we provide a type called
:class:`FoldsView`, which associates a *data container* with a
given sequence of assignments.

.. class:: FoldsView <: DataView <: AbstractVector

   A vector-like representation of applying a repartitioning
   strategy to a specific data container. It is used to associate
   a data container with appropriate assignments, and will act as
   a lazy view, that allows the data to be treated as a sequence
   of folds. As such it does not copy any data.

   :class:`FoldsView` is a subtype of ``AbstractArray`` and
   as such supports the appropriate interface. Each individual
   element (accessible via ``getindex``) is a tuple of two
   subsets of the data container; a training- and a validation
   subset.

   .. attribute:: data

      The object describing the data source of interest. Can be
      of any type as long as it implements the :ref:`container`
      interface.

   .. attribute:: train_indices

      Vector of integer vectors containing the sequences of
      assignments for the *training* subsets. This means that
      each element of this vector is a vector of
      observation-indices valid for ``data``. The length of this
      vector must match ``val_indices``, and denotes the number
      of folds.

   .. attribute:: val_indices

      Vector of integer vectors containing the sequences of
      assignments for the *validation* subsets. This means that
      each element of this vector is a vector of
      observation-indices valid for ``data``. The length of this
      vector must match ``train_indices``, and denotes the number
      of folds.

   .. attribute:: obsdim

      If defined for the type of data, ``obsdim`` can be used to
      specify which dimension of ``data`` denotes the
      observations. Should be ``ObsDim.Undefined`` if not
      applicable.

The purpose of :class:`FoldsView` is to apply a precomputed
sequence of assignments to some data container in a convenient
manner. By itself, :class:`FoldsView` is agnostic to any
particular repartitioning- or resampling strategy. Instead, the
assignments, ``train_indices`` and ``val_indices``, need to be
precomputed by such a strategy and then passed to
:func:`FoldsView` with a concrete data container. The resulting
object can then be queried for its individual folds using
``getindex``, or alternatively, simply iterated over.

.. function:: FoldsView(data, train_indices, val_indices, [obsdim]) -> FoldsView

   Create a :class:`FoldsView` for the given `data` container.
   The number of folds is denoted by the length of
   `train_indices`, which must be equal to the length of
   `val_indices`.

   Note that the number of observations in `data` is expected to
   match the number of observations that the given assignments
   were designed for.

   :param data: The object representing a data container.

   :param AbstractVector train_indices: \
        Vector of integer vectors. It denotes the sequence of
        training assignments (i.e. the indices of the training
        subsets).

   :param AbstractVector val_indices: \
        Vector of integer vectors. It denotes the sequence of
        validation assignments (i.e. the indices of the
        validation subsets)

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

To get a better feeling of how exactly :class:`FoldsView` works,
let us consider the following toy data container ``X``. We will
generate this data in such a way, that it is easy to see where
each observation ends up after applying our partitioning
strategy. To keep it simple let's say it has 10 observations with
2 features each.

.. code-block:: jlcon

   julia> X = hcat(1.:10, 11.:20)' # generate toy data
   2×10 Array{Float64,2}:
     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

First we need to compute appropriate assignments that are
applicable to our data container ``X``. Ideally these assignments
should follow some repartitioning strategy. For this example we
will use :func:`kfolds`, which we introduced in a previous
section. In particular we will compute the sequence of
assignments for a 5-fold repartitioning.

.. code-block:: jlcon

   julia> train_idx, val_idx = kfolds(10, 5);

   julia> train_idx
   5-element Array{Array{Int64,1},1}:
    [3,4,5,6,7,8,9,10]
    [1,2,5,6,7,8,9,10]
    [1,2,3,4,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> val_idx
   5-element Array{UnitRange{Int64},1}:
    1:2
    3:4
    5:6
    7:8
    9:10

Now that we have appropriate assignments, we can use
:class:`FoldsView` to apply those to our data container ``X``.
Note that since :class:`FoldsView` is designed to act as a
"view", it won't actually copy any data from ``X``, instead each
"fold" will be a tuple of two ``SubArray`` into ``X``.

.. code-block:: jlcon

   julia> folds = FoldsView(X, train_idx, val_idx) # output reformated for readability
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}},Array{Float64,2},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

   julia> train, val = folds[2]; # access second fold

   julia> train
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
     1.0   2.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  15.0  16.0  17.0  18.0  19.0  20.0

   julia> val
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
     3.0   4.0
    13.0  14.0

As we can see in the above example, each element of ``folds`` is
a tuple of two data subsets. More specifically, since our data
container ``X`` is an ``Array``, each tuple element is a
``SubArray`` into some part of ``X``.

Similar to most other functions defined by this package, you can
use the optional parameter ``obsdim`` to specify which dimension
of ``data`` denotes the observations. If that concept does not
make sense for the type of ``data`` it can simply be omitted. For
example, the following code shows how we could work with a
transposed version of ``X``, where the first dimension enumerates
the observations.

.. code-block:: jlcon

   julia> folds = FoldsView(X', train_idx, val_idx, obsdim=1) # note the transpose
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false},SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}},Array{Float64,2},LearnBase.ObsDim.Constant{1},Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 13.0; 4.0 14.0; … ; 9.0 19.0; 10.0 20.0], [1.0 11.0; 2.0 12.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [3.0 13.0; 4.0 14.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [5.0 15.0; 6.0 16.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [7.0 17.0; 8.0 18.0])
    ([1.0 11.0; 2.0 12.0; … ; 7.0 17.0; 8.0  18.0], [9.0 19.0; 10.0 20.0])

   julia> train, val = folds[2]; # access second fold

   julia> train
   8×2 SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}:
     1.0  11.0
     2.0  12.0
     5.0  15.0
     6.0  16.0
     7.0  17.0
     8.0  18.0
     9.0  19.0
    10.0  20.0

   julia> val
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}:
    3.0  13.0
    4.0  14.0

It is also possible to link multiple different data containers
together on an per-observation level. This way they can be
repartitioned as one coherent unit. To do that, simply put all
the relevant data container into a single ``Tuple``, before
passing it to :func:`FoldsView`.

.. code-block:: jlcon

   julia> y = collect(1.:10) # generate a toy target vector
   10-element Array{Float64,1}:
     1.0
     2.0
     3.0
     ⋮
     8.0
     9.0
    10.0

   julia> folds = FoldsView((X, y), train_idx, val_idx); # note the tuple

   julia> (train_x, train_y), (val_x, val_y) = folds[2]; # access second fold

   julia> val_x
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
     3.0   4.0
    13.0  14.0

   julia> val_y
   2-element SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}:
    3.0
    4.0

It is worth pointing out, that the tuple elements (i.e. data
container) need not be of the same type, nor of the same shape.
You can observe this in the code above, where ``X`` is a
``Matrix`` while ``y`` is a ``Vector``. Note, however, that all
tuple elements must be data containers themselves. Furthermore,
they all must contain the same exact number of observations.

While it is useful and convenient to be able to access some
specific fold using the ``getindex`` syntax sugar (e.g.
``folds[2]``), :class:`FoldsView` can also be iterated over (just
like any other ``AbstractVector``). In fact, this is the main
intention behind its design, because it allows you to
conveniently loop over all folds.

.. code-block:: julia

   julia> for (X_train, X_val) in FoldsView(X, train_idx, val_idx)
              println(X_val) # do something useful here instead
          end
   [1.0 2.0; 11.0 12.0]
   [3.0 4.0; 13.0 14.0]
   [5.0 6.0; 15.0 16.0]
   [7.0 8.0; 17.0 18.0]
   [9.0 10.0; 19.0 20.0]

So far we showed how to use the low-level API to perform a
repartitioning strategy on some data container. This was a
two-step process. First we had to compute the assignments, and
then we had to apply those assignment to some data container
using the type :class:`FoldsView`. In the rest of this document
we will see how to do the same tasks in just one single step by
using the high-level API.

.. _k_folds:

K-Folds for Data Container
-----------------------------

Let us revisit the idea behind a :math:`k`-folds repartitioning
strategy, which we introduced in the beginning of this document.
Conceptually, :math:`k`-folds divides the given data container
into :math:`k` roughly equal-sized parts. Each part will serve as
validation set once, while the remaining parts are used for
training at that stage. This results in :math:`k` different
partitions of the same data.

We have already seen how to compute the assignments of a
:math:`k`-folds scheme manually, and how to apply those to a data
container using the type :class:`FoldsView`. We can do both those
steps in just one single swoop by passing the data container to
:func:`kfolds` directly.

.. function:: kfolds(data, [k = 5], [obsdim]) -> FoldsView

   Repartition a `data` container `k` times using a `k`-folds
   strategy and return the sequence of folds as a lazy
   :class:`FoldsView`. The resulting :class:`FoldsView` can then
   be indexed into or iterated over. Either way, only data
   subsets are created. That means that no actual data is copied
   until :func:`getobs` is invoked.

   In the case that the number of observations in `data` is not
   dividable by the specified `k`, the remaining observations
   will be evenly distributed among the parts. Note that there is
   no random assignment of observations to parts, which means
   that adjacent observations are likely part of the same
   validation subset.

   :param data: The object representing a data container.

   :param Integer k: \
        Optional. The number of folds to compute. Can be
        specified as positional argument or as keyword argument.
        A general rule of thumb is to use either ``k = 5`` or ``k
        = 10``. Must be within the range ``2:nobs(data)``.
        Defaults to ``k = 5``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

To visualize what exactly :func:`kfolds` does, let us consider
the following toy data container ``X``. We will generate this
data in such a way, that makes it easy to see where each
observation ends up after we apply the partitioning strategy to
it. To keep it simple let’s say it has 10 observations with 2
features each.

.. code-block:: jlcon

   julia> X = hcat(1.:10, 11.:20)' # generate toy data
   2×10 Array{Float64,2}:
     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

Now that we have a data container to work with, we can pass it to
the function :func:`kfolds` to create a view of the data that
lets us treat it as a sequence of distinct partitions/folds.

.. code-block:: jlcon

   julia> folds = kfolds(X, k = 5) # output reformated for readability
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}},Array{Float64,2},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

We can now query any individual fold using the typical indexing
syntax. For instance, the following code snippet shows the
training- and validation subset of the third fold.

.. code-block:: jlcon

   julia> train, val = folds[3]; # access third fold

   julia> train
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
     1.0   2.0   3.0   4.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  17.0  18.0  19.0  20.0

   julia> val
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
     5.0   6.0
    15.0  16.0

Note how ``train`` and ``val`` are of type ``SubArray``, which
means that their content isn't actually a copy from ``X``.
Instead, they serve as a view into the original data container
``X``. For more information about on that topic take a look at
:ref:`subsets`.

If instead of a view you would like to have the folds as actual
``Array``, you can use :func:`getobs` on the :class:`FoldsView`.
This will trigger :func:`getobs` on each subset and return the
result as a ``Vector``.

.. code-block:: jlcon

   julia> getobs(folds) # output reformated for readability
   5-element Array{Tuple{Array{Float64,2},Array{Float64,2}},1}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

   julia> fold_3 = getobs(folds, 3)
   ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [5.0 15.0; 6.0 16.0])

   julia> typeof(fold_3)
   Tuple{Array{Float64,2},Array{Float64,2}}

You can use the optional parameter ``obsdim`` to specify which
dimension of data denotes the observations. It can be specified
as positional argument (which is type-stable) or as a more
convenient keyword argument. For instance, the following code
shows how we could work with a transposed version of ``X``, where
the first dimension enumerates the observations.

.. code-block:: jlcon

   julia> folds = kfolds(X', 5, ObsDim.First()); # equivalent to below, but typesable

   julia> folds = kfolds(X', k = 5, obsdim = 1) # note the transpose
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false},SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}},Array{Float64,2},LearnBase.ObsDim.Constant{1},Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 13.0; 4.0 14.0; … ; 9.0 19.0; 10.0 20.0], [1.0 11.0; 2.0 12.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [3.0 13.0; 4.0 14.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [5.0 15.0; 6.0 16.0])
    ([1.0 11.0; 2.0 12.0; … ; 9.0 19.0; 10.0 20.0], [7.0 17.0; 8.0 18.0])
    ([1.0 11.0; 2.0 12.0; … ; 7.0 17.0;  8.0 18.0], [9.0 19.0; 10.0 20.0])

It is also possible to call :func:`kfolds` with multiple data
containers wrapped in a ``Tuple``. Note, however, that all data
containers must have the same total number of observations. Using
a tuple this way will link those data containers together on a
per-observation basis.

.. code-block:: jlcon

   julia> y = collect(1.:10) # generate a toy target vector
   10-element Array{Float64,1}:
     1.0
     2.0
     3.0
     ⋮
     8.0
     9.0
    10.0

   julia> folds = kfolds((X, y), k = 5); # note the tuple

   julia> (train_x, train_y), (val_x, val_y) = folds[2]; # access second fold

For more information and additional examples on what you can do
with the result of :func:`kfolds`, take a look at
:ref:`foldsview`.

Leave-Out for Data Container
--------------------------------

Recall how we motivated leave-:math:`p`-out as a different way to
think about :math:`k`-folds. Instead of specifying the number of
folds :math:`k` directly, we specify how many observations of the
given data container should be in each validation subset.

Similar to :func:`kfolds`, we provide a method for
:func:`leaveout` that allows it to be invoked with a data
container. This method serves as a convenience layer that will
return an appropriate :class:`FoldsView` of the given data for
you.

.. function:: leaveout(data, [size = 1], [obsdim]) -> FoldsView

   Repartition a `data` container using a k-fold strategy, where
   ``k`` is chosen in such a way, that each validation subset of
   the computed folds contains roughly `size` observations. The
   resulting sequence of folds is then returned as a lazy
   :class:`FoldsView`, which can be index into or iterated over.
   Either way, only data subsets are created. That means no
   actual data is copied until :func:`getobs` is invoked.

   :param data: The object representing a data container.

   :param Integer size: \
        Optional. The desired number of observations in each
        validation subset. Can be specified as positional
        argument or as keyword argument. Defaults to ``size =
        1``, which results in a "leave-one-out" partitioning.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Let us again consider the toy feature-matrix ``X`` from before.
We can pass it to the function :func:`leaveout` to create a view
of the data. This "view" is represented as a :class:`FoldsView`
which lets us treat it is as a sequence of distinct
partitions/folds.

.. code-block:: jlcon

   julia> X = hcat(1.:10, 11.:20)' # generate toy data
   2×10 Array{Float64,2}:
     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

   julia> folds = leaveout(X, size = 2) # output reformated for readability
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}},Array{Float64,2},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

We can now query any individual fold using the typical indexing
syntax. Additionally, the function :func:`leavout` supports all
the signatures of :func:`kfolds`. For more information and
additional examples on what you can do with the result of
:func:`leaveout`, take a look at :ref:`foldsview`.
