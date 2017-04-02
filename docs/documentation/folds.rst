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
other words, the purpose of the type :class:`KFolds` is to
provide an abstraction to randomly partition some data set into
:math:`k` disjoint folds. :class:`KFolds` is best utilized as an
iterator. If used as such, the data set will be split into
different training and test portions in :math:`k` different and
unqiue ways, each time using a different fold as the
validation-/test-set.

The following code snippets showcase how the function
:func:`kfolds` could be utilized:

TODO: example KFolds

.. note:: The sizes of the folds may differ by up to 1
   observation depending on if the total number of observations
   is dividable by :math:`k`.

