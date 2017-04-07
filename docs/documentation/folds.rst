.. _folds:

Repartitioning Strategies
================================

Most non-trivial machine learning experiments require some form
of model tweaking *prior* to training. A particularly common
scenario is when the model (or algorithm) has hyper parameters
that need to be specified manually. The process of searching for
suitable hyper parameters is just one part of what we call *model
selection*.

If model selection is part of the experiment, then it is quite
likely that a simple train/test split will not be effective
enough to achieve results that are representative for new or
unseen data. The reason for this is subtle, but very important.
If the hyper parameters are chosen based on how well the model
performs on the test set, then information about the test set is
actively fed back into the model. This is because the test set is
used several times *and* decisions are made based on what was
observed. In other words: the test set participates in an aspect
of the training process, namely the model selection.
Consequently, the results on the test set become less
representative for the expected results on new, unseen data. To
avoid causing this kind of manual overfitting, one should instead
somehow make use of the training set for such a model selection
process, while leaving the test set out of it completely. Luckily
this can be done quite effectively using a repartitioning
strategy, such as a :math:`k`-folds, and using the various
partitions to perform cross-validation.

We will start by discussing the terminology we use, and - more
importantly - how the various terms are used in the context of
this package. The rest of this document will then focus on how
these concepts exposed to the user. We will start by introducing
some low-level helper methods for computing the required
assignments sequences. We will then use those assignments to
motivate a type called :class:`FoldsView`, which can be
configured to represent almost any kind of repartitioning
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
  particular configuration of assigning the observations from
  some data container to multiple disjoined subsets. In contrast
  to the formal definition in mathematics, we do allow the same
  observation to occur multiple times in the *same* subset.

  For instance the function :func:`splitobs` creates a single
  partition in the form of a tuple. More concretely, the
  following code snipped creates a partition with two subsets
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
  code snippet computes 3 partitions/folds for such a strategy on
  a random toy data-vector ``y`` that has 5 observations in it.

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
  that is valid for any data container of size :math:`N`, by
  using indices from the set :math:`\{1,2,...,N\}`. For instance,
  if a single partition should consist of two subsets, then the
  corresponding assignment (singular), is made up of two vectors
  of indices, each vector describing the content of one subset.
  Because of this, it is also fair to think about the result of a
  repartitioning strategy as two sequences, one for the *training
  assignments* and a corresponding sequence for the *validation
  assignments*.

  To give a concrete example of such assignment sequences,
  consider the result of calling ``kfolds(6, 3)`` (see code
  below). It will compute the training assignments ``train_idx``
  and the validation assignments ``val_idx`` for a 3-fold
  repartitioning strategy that is applicable to any data
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

Computing K-Fold Indices
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
computed assignment indices that are applicable to *any* data set
that has exactly 10 observations in it. The important thing to
note here is that while the indices in ``train_idx`` overlap, the
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
            length `k`, where each element is also a vector. The
            first vector represents the sequence of training
            assignments, and the second vector the sequence
            of validation assignments.

Invoking :func:`leaveout` with an integer as first parameter will
compute the sequence of assignments for a :math:`k`-folds
repartitioning strategy. For example, the following code will
assign the indices of 10 observations to as many partitions as it
takes such that every validation partition contains approximately
2 observations.

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

The FoldsView Type
-----------------------

So far we focused on just computing the sequence of assignments
for various repartition strategies, without any regard to an
actual data set. Instead, we just specified the total number of
observations. Naturally that is only one part of the puzzle.
After all what we really care about is the repartitioning of an
actual data set. To that end we provide a type called
:class:`FoldsView`, which associates a *data container* with a
given sequence of assignments.

.. class:: FoldsView <: AbstractVector

   A vector-like representation of applying a repartitioning
   strategy to a specific data container.

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
      assignments for the *test* subsets. This means that each
      element of this vector is a vector of observation-indices
      valid for ``data``. The length of this vector must match
      ``train_indices``, and denotes the number of folds.

   .. attribute:: obsdim

      If defined for the type of data, ``obsdim`` can be used to
      specify which dimension of ``data`` denotes the
      observations. Should be ``ObsDim.Undefined`` if not
      applicable.

The purpose of :class:`FoldsView` is to apply a precomputed
sequence of assignment indices to some data container in a
convenient manner. By itself, :class:`FoldsView` is agnostic to
any particular repartitioning- or resampling strategy. Instead
the assignments, ``train_indices`` and ``val_indices``, need to
be precomputed by such a strategy and then passed to
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
        training assignments. I.e. the indices of the training
        subsets

   :param AbstractVector val_indices: \
        Vector of integer vectors. It denotes the sequence of
        validation assignments. I.e. the indices of the
        validation subsets

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

To get a better feeling of how exactly :class:`FoldsView` works,
let us consider the following toy data container ``X``. We will
generate it in such a way, that it is easy to see where each
observation ends up in our partitioning strategy. To keep it
simple let's say it has 10 observations with 2 features each.

.. code-block:: jlcon

   julia> X = hcat(1.:10, 11.:20)' # generate toy data
   2×10 Array{Float64,2}:
     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

First we need to compute appropriate assignments that are
applicable to our data set. Ideally these assignments should
follow some repartitioning strategy. For this example we will use
:func:`kfolds`, which we introduced in a previous section. In
particular we will compute the sequence of assignments for a
5-fold repartitioning.

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
Note that since the type is designed as a "view", it won't
actually copy any data from ``X``, instead each "fold" will
be a ``SubArray`` into ``X``.

.. code-block:: jlcon

   julia> folds = FoldsView(X, train_idx, val_idx) # output reformated for readability
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}},Array{Float64,2},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

   julia> train_2, val_2 = folds[2]; # access second fold

   julia> train_2
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
     1.0   2.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  15.0  16.0  17.0  18.0  19.0  20.0

   julia> val_2
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

   julia> train_2, val_2 = folds[2]; # access second fold

   julia> train_2
   8×2 SubArray{Float64,2,Array{Float64,2},Tuple{Array{Int64,1},Colon},false}:
     1.0  11.0
     2.0  12.0
     5.0  15.0
     6.0  16.0
     7.0  17.0
     8.0  18.0
     9.0  19.0
    10.0  20.0

   julia> val_2
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

   julia> (train_x, train_y), (val_x, val_y) = folds[2]; # access second partitioning

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

So far we showed how to access some specific fold using the
``getindex`` syntax sugar (e.g. ``folds[2]``). Because
:class:`FoldsView` is a subtype of ``AbstractVector``, it can
also be iterated over. In fact, this is the main intention behind
its design.

.. code-block:: julia

   julia> for (X_train, X_val) in FoldsView(X, train_idx, val_idx)
              println(X_val) # do something useful here instead
          end
   [1.0 2.0; 11.0 12.0]
   [3.0 4.0; 13.0 14.0]
   [5.0 6.0; 15.0 16.0]
   [7.0 8.0; 17.0 18.0]
   [9.0 10.0; 19.0 20.0]

K-Folds for Data Container
-----------------------------

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.

Leave-Out for Data Container
--------------------------------

