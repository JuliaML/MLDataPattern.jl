.. MLDataPattern.jl documentation master file, created by
   sphinx-quickstart on Fri Mar 31 14:06:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLDataPattern.jl's documentation
=====================================

This package represents a community effort to provide a native
and generic `Julia <http://julialang.org>`_ implementation for
commonly used **data access pattern** in Machine Learning. As
such, it is a part of the `JuliaML <https://github.com/JuliaML>`_
ecosystem.

In contrast to other data-centered Julia packages, the focus of
MLDataPattern is specifically on functionality utilized in a
Machine Learning context. This includes *shuffling*,
*partitioning*, and *resampling* data sets of various types and
origin. More importantly, this package was designed around the
core premise of allowing any user-defined type to serve as a
custom data source and/or access pattern in a first class manner.


Where to begin?
----------------

If this is the first time you consider using MLDataPattern for
your machine learning related experiments or packages, make sure
to check out the "Getting Started" section; specifically "How to
...?", which lists some of most common scenarios and links to the
appropriate places that should guide you on how to approach these
scenarios using the functionality provided by this or other
packages.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted

Introduction and Motivation
-----------------------------

If you are new to Machine Learning in Julia, or are simply
interested in how and why this package works the way it works,
feel free to take a look at the following documents. There we
discuss the problem of data-partitioning itself and what
challenges it entails. Further we will provide some insight on
how this package approaches the task conceptually.

.. toctree::
   :maxdepth: 2

   introduction/motivation
   introduction/design

Using MLDataPattern.jl
---------------------

The main design principle behind this package is based on the
assumption that the data source a user is working with, is likely
of some user-specific custom type. That said, there was also a
lot of attention put into first class support for those types
that are most commonly employed to represent the data of
interest, such as ``Array``.

The first topic we will cover is about **data containers**. These
represent a large subgroup of data sources, that all know how
many observations they contain, as well as how to access specific
observation(s). As such they are the most flexible kind of data
sources and will thus be at the heart of most of the subsequent
sections. To start off, we will discuss what makes some type a
data container and what that term entails.

.. toctree::
   :maxdepth: 2

   documentation/container

Once we understand what data containers are and how they can be
interacted with, we can introduce more interesting behaviour on
top of them. The most enabling of them all is the idea of a
**data subset**. A data subset is in essence just a lazy
representation of a specific sequence of observations from a data
container, and itself again a data container. What that means and
why that is useful will be discussed in detail in the following
section.

.. toctree::
   :maxdepth: 3

   documentation/datasubset

At this point we know what data container and data subsets are.
In particular we discussed how we can split our data container
into disjoint subsets. We have even seen how we can use tuples to
link multiple data container together on a per-observation level.
While we mentioned that this is particular useful for labeled
data, we did not really elaborate on what that means. In order to
change that, we will spend the next section solely on working
with data container that have **targets**. This will put us into
the realm of supervised learning. We will see how we can work
with labeled data container and what special functionality is
available for them.

.. TODO Labeled Data Sources

Now that we have discussed all the basics, we can start to cover
some of the more advanced topic. A particularly important aspect
of modern Machine Learning is what is known as *model selection*.
Most of the time, this boils down to choosing appropriate
hyper-parameters for the model one is working with. To avoid bias
in this selection process, it is quite common to employ some kind
of **repartitioning strategy** on the training data. One of the
most famous of these strategies is :math:`k`-folds cross
validation. Of course the partitioning is just one part of such a
model selection process, since we still have to compute and
compare the performance somehow. However, it is an important step
that is needed to make the most of the available data. So
important in fact, that we will spend a whole section on it.

.. toctree:: :maxdepth: 2

   documentation/folds


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

