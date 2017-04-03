.. Folds and Resampling Strategies

.. _folds:

Repartitioning Strategies
================================

Most non-trivial machine learning experiments require some form
of model tweaking prior to training. A particularly common
scenario is when the model (or algorithm) has hyper parameters,
which need to be specified manually. If that is the case, then
chances are that a simple train/test split won't be enough
anymore. At least not if we want to be confident in our results.
The reason for this is subtle, but very important. If you choose
your hyper parameters based on how well your model performs on
the test set, then you basically feed back information about your
test set into your model. This is because you use your test set
several times, and make decisions based on what you see.
Consequently, the results on your test set become less
representative for the expected results on new, unseen data.

The rest of thus document will focus on how this package
approaches the task of repartitioning. We will start by
introducing some low level helper functions for computing
indices-assignments. Then we will introduce a type called
:class:`FoldsView`, which can be configured to represent almost
any type of repartitioning or resampling strategy. After
introducing these basics, we will introduce the high-level
functions that serve as a convenience layer around
:class:`FoldsView`.

Computing K-Fold Indices
--------------------------

A particularly popular strategy for model selection is *k-fold
cross-validation*, in which the data set gets partitioned into
:math:`k` folds. Each model is fit :math:`k` times, while each
time a different fold is left out during training, and is instead
used as a validation set. The validation performance of the
:math:`k` instances of the model is then averaged over all folds
and reported as the performance for the particular set of
hyper parameters.

Computing Leave-N-Out Indices
--------------------------------

The FoldsView Type
-----------------------

K-Folds for Data Container
-----------------------------

Leave-N-Out for Data Container
--------------------------------

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.

