.. Folds and Resampling Strategies

.. _folds:

Repartitioning Strategies
================================

Most non-trivial machine learning experiments require some form
of model tweaking prior to training. A particularly common
scenario is when the model (or algorithm) has hyper parameters
that need to be specified manually. We call the process of
finding suitable hyper parameters *model selection*. If model
selection is involved, then it is quite likely, that a simple
train/test split is not enough to achieve results that are
representative for new and unseen data. The reason for this is
subtle, but very important. If you choose your hyper parameters
based on how well your model performs on the test set, then you
basically feed back information about your test set into your
model. This is because you use your test set several times and
make decisions based on what you see. In a way this means hat the
test set participates in an aspect of the training process,
namely the model selection. Consequently, the results on your
test set become less representative for the expected results on
new, unseen data. To avoid causing this kind of manual
overfitting, we should instead somehow make use of the training
set for such a model selection process, while leaving the test
set out of it completely. Luckily this can be done quite
effectively using a repartitioning strategy, such as
:math:`k`-folds.

The rest of this document will focus on how this package
approaches the task of repartitioning. We will start by
introducing some low-level helper methods for computing
indices-assignments. We will then use those assignments to
motivate a type called :class:`FoldsView`, which can be
configured to represent almost any type of repartitioning or
resampling strategy. After discussing these basics, we will
introduce the high-level methods that serve as a convenience
layer around :class:`FoldsView`.

Computing K-Fold Indices
--------------------------

A particularly popular strategy for model selection is *k-fold
cross-validation*; the first step of which is partitioning the
data set into :math:`k` folds. Each model is fit :math:`k` times,
while each time a different fold is left out during training, and
is instead used as a validation set to compute the metric of
interest. The validation results of the :math:`k` instances of
the model are then averaged over all folds and reported as the
performance for the particular set of hyper parameters.

Before we go into details about the partitioning or, later, the
validation aspects, let us first consider how to compute the
folds. In particular how to compute the indices assignments,
which are the assignments of observation-indices to the various
folds. For that purpose we provide a helper method for the
function :func:`kfolds`.

.. function:: kfolds(n, [k = 5]) -> Tuple

   Compute the train/test indices for `k` folds for `n`
   observations, and return them in the form of two vectors. The
   first vector contains the indices for the training folds, and
   the second vector the indices for test folds respectively.

   Each observation is assigned to the test indices once (and
   only once). Note that there is no random assignment of
   observations to folds, which means that adjacent observations
   are likely part of the same fold.

   *Note*: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   `n` is dividable by `k`.

   :param Integer n: Total number of observations to compute the
                     folds for.

   :param Integer k: Optional. The number of folds to compute. A
                     general rule of thumb is to use either ``k =
                     5`` or ``k = 10``. Must be within the range
                     ``2:n``. Defaults to ``k = 5``.

   :return: A ``Tuple`` of two ``Vector{Int}``. Both vectors are
            of length `k`. The first vector contains the
            indices-assignments for the training folds, and the
            second vector the corresponding indices-assignments
            for the test folds

Invoking :func:`kfolds` with an integer as first parameter - as
outlined above - will compute the indices assignments for a
:math:`k`-folds repartitioning strategy. For example, the
following code will partition the indices of 10 observations to 4
folds.

.. code-block:: jlcon

   julia> train_idx, test_idx = kfolds(10, 4); # assign 10 observation to 4 folds

   julia> train_idx
   4-element Array{Array{Int64,1},1}:
    [4,5,6,7,8,9,10]
    [1,2,3,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> test_idx
   4-element Array{UnitRange{Int64},1}:
    1:3
    4:6
    7:8
    9:10

As we can see, there is no actual data set involved yet. We just
computed indices assignments that are applicable to *any* data
set that has exactly 10 observations in it. The important thing
to note here is that while the indices in ``train_idx`` overlap,
the indices in ``test_idx`` do not.

Computing Leave-Out Indices
--------------------------------

A different way to think about a :math:`k`-folds partitioning
strategy is in terms of the test fold size. Instead of specifying
the number of folds directly, we specify how many observations we
would like to be in each test fold. While the resulting indices
assignment is equivalent to a :math:`k`-folds scheme, it is
sometimes referred to as *leave-out partitioning*. A particularly
common version of which is leave-one-out, where we set the test
fold size to 1 observation.

.. function:: leaveout(n, [size = 1]) -> Tuple

   Compute the train/test indices for ``k ≈ n/size`` folds for
   `n` observations and return them in the form of two vectors.
   The first vector contains the indices for the training folds,
   and the second vector the indices for test folds respectively.

   Each observation is assigned to the test indices once (and
   only once). Furthermore, each test fold will have either
   `size` or `size` + 1 observations assigned to it.

   Note that there is no random assignment of observations to
   folds, which means that adjacent observations are likely part
   of the same fold.

   :param Integer n: Total number of observations to compute the
                     folds for.

   :param Integer size: The desired number of observations in
                        each test fold.

   :return: A ``Tuple`` of two ``Vector{Int}``. Both vectors are
            of length ``k``. The first vector contains the
            indices-assignments for the training set, and the
            second vector the corresponding indices-assignments
            for the test set

Invoking :func:`leaveout` with an integer as first parameter will
compute the indices assignments for a :math:`k`-folds
repartitioning strategy. For example, the following code will
assign the indices of 10 observations to as many folds as it
takes such that every test fold contains around 2 observations,
and every observation is part of a test fold once.

.. code-block:: jlcon

   julia> train_idx, test_idx = leaveout(10, 2);

   julia> train_idx
   5-element Array{Array{Int64,1},1}:
    [3,4,5,6,7,8,9,10]
    [1,2,5,6,7,8,9,10]
    [1,2,3,4,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> test_idx
   5-element Array{UnitRange{Int64},1}:
    1:2
    3:4
    5:6
    7:8
    9:10

Just like before, there is no actual data set involved here. We
just computed indices-assignments applicable to *any* data set
that has exactly 10 observations in it. Note that for the above
example the result is equivalent to calling ``kfolds(10, 5)``.

The FoldsView Type
-----------------------

So far we have seen how to compute indices assignments without
involving any actual data set at all. Naturally that is only one
part of the puzzle. After all what we really care about is
partitioning an actual data set. To that end we provide a type
called :class:`FoldsView`, which associates a *data container*
with a given indices assignment.

.. class:: FoldsView <: AbstractVector

   Create a vector-like representation of a given data container,
   where each individual element is a tuple of two data subsets;
   a training and a test fold. The first element of each tuple
   corresponds to the indices stored in the corresponding element
   of ``train_indices``, while the second element of each tuple
   corresponds to ``test_indices``.

   .. attribute:: data

      The object describing the data source of interest. Can
      be of any type as long as it implements the data container
      interface.

   .. attribute:: train_indices

      Vector of integer vectors containing the
      indices-assignments of each *training* fold. This means that
      each element of this vector is a vector of observation
      indices. The length of this vector must match
      ``test_indices``, and denotes the number of folds.

   .. attribute:: test_indices

      Vector of integer vectors containing the
      indices-assignments of each *test* fold. This means that each
      element of this vector is a vector of observation indices.
      The length of this vector must match ``train_indices``, and
      denotes the number of folds.

   .. attribute:: obsdim

      If defined for the type of data, ``obsdim`` can be used to
      specify which dimension of ``data`` denotes the
      observations. Should be ``ObsDim.Undefined`` if not
      applicable.

The purpose of :class:`FoldsView` is to apply precomputed fold
indices to some data container in a convenient manner. By itself,
:class:`FoldsView` is agnostic to any particular repartitioning
or resampling strategy (such as math:`k`-folds). Instead the fold
assignment indices, ``train_indices`` and ``test_indices``, need
to be precomputed by such a strategy and then passed to
:func:`FoldsView` with a concrete data container. The resulting
object can then be queried for its individual splits using
``getindex``, or simply iterated over.

.. function:: FoldsView(data, train_indices, test_indices, [obsdim]) -> FoldsView

   Create a :class:`FoldsView` for the given `data` container.
   The number of folds is denoted by the length of
   `train_indices`, which must be equal to the length of
   `test_indices`.

   Note that the number of observations in `data` is expected to
   match the number of observations that the given indices were
   designed for.

   :class:`FoldsView` is a subtype of ``AbstractArray`` and as
   such supports the appropriate interface.

   :param data: The object representing a data container.

   :param AbstractVector train_indices:
        Indices-assignments for the training folds.

   :param AbstractVector test_indices:
        Indices-assignments for the test folds.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        typestable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

To get a better feeling of how exactly :class:`FoldsView` works,
let us consider the following toy data container ``X``. We will
create it in such a way that it is easy to see where each
observation ends up in our partitioning strategy. To keep it
simple let's say it has 10 observations with 2 features each.

.. code-block:: jlcon

   julia> X = hcat(1.:10, 11.:20)'
   2×10 Array{Float64,2}:
     1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    11.0  12.0  13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

First we need to compute an appropriate indices-assignment for
our data set using some repartitioning strategy. Let's use
:func:`kfolds` for this, which we introduced in a previous
section. In particular we will compute the assignments for 5
folds.

.. code-block:: jlcon

   julia> train_idx, test_idx = kfolds(10, 5);

   julia> train_idx
   5-element Array{Array{Int64,1},1}:
    [3,4,5,6,7,8,9,10]
    [1,2,5,6,7,8,9,10]
    [1,2,3,4,7,8,9,10]
    [1,2,3,4,5,6,9,10]
    [1,2,3,4,5,6,7,8]

   julia> test_idx
   5-element Array{UnitRange{Int64},1}:
    1:2
    3:4
    5:6
    7:8
    9:10

Now that we have appropriate assignments, we can use
:class:`FoldsView` to apply them to our data container ``X``.
Note that since it is a "view", we won't actually copy any data
from ``X``, instead each "partition" will be a ``SubArray`` into
``X``.

.. code-block:: jlcon

   julia> folds = FoldsView(X, train_idx, test_idx) # output reformated for readability
   5-element MLDataPattern.FoldsView{Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false},SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}},Array{Float64,2},LearnBase.ObsDim.Last,Array{Array{Int64,1},1},Array{UnitRange{Int64},1}}:
    ([3.0 4.0 … 9.0 10.0; 13.0 14.0 … 19.0 20.0], [1.0  2.0; 11.0 12.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [3.0  4.0; 13.0 14.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [5.0  6.0; 15.0 16.0])
    ([1.0 2.0 … 9.0 10.0; 11.0 12.0 … 19.0 20.0], [7.0  8.0; 17.0 18.0])
    ([1.0 2.0 … 7.0  8.0; 11.0 12.0 … 17.0 18.0], [9.0 10.0; 19.0 20.0])

   julia> train_1, test_1 = folds[1]; # access first fold

   julia> train_1
   2×8 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,Array{Int64,1}},false}:
     3.0   4.0   5.0   6.0   7.0   8.0   9.0  10.0
    13.0  14.0  15.0  16.0  17.0  18.0  19.0  20.0

   julia> test_1
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
     1.0   2.0
    11.0  12.0

As we can see in the above example, each element of ``folds`` is
a tuple of two data subsets. More specifically, since our data
container ``X`` is an ``Array``, each tuple element is a
``SubArray`` into some part of ``X``.

K-Folds for Data Container
-----------------------------

Leave-Out for Data Container
--------------------------------

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.

