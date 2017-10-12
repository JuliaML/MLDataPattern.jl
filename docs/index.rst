.. MLDataPattern.jl documentation master file, created by
   sphinx-quickstart on Fri Mar 31 14:06:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLDataPattern.jl's documentation
=====================================

This package represents a community effort to provide a native
and generic `Julia <http://julialang.org>`_ implementation for
commonly used **data access pattern** in Machine Learning. Most
notably it provides a number of pattern for *shuffling*,
*partitioning*, and *resampling* data sets of various types and
origin. At its core, the package was designed around the key
requirement of allowing any user-defined type to serve as a
custom data source and/or access pattern in a first class manner.

In contrast to other data-related Julia packages, the focus of
MLDataPattern is specifically on functionality utilized in a
machine learning context. As such, it is a part of the `JuliaML
<https://github.com/JuliaML>`_ ecosystem.

Where to begin?
----------------

If this is the first time you consider using MLDataPattern for
your machine learning related experiments or packages, make sure
to check out the "Getting Started" section. It will provide a
very condensed overview of all the topics outlined below. If you
are looking to perform some specific task, then take a look at
“How to ...?”, which lists some of most common scenarios and
links to the appropriate places that should guide you on how to
approach these scenarios using the functionality provided by this
or other packages.

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
container, the sequence itself being another data container. What
that means and why that is useful will be discussed in detail in
the following section.

.. toctree::
   :maxdepth: 3

   documentation/datasubset

By this point we know what data containers and data subsets are.
In particular, we discussed how we can split our data container
into disjoint subsets. We have even seen how we can use tuples to
link multiple data container together on a per-observation level.
While we mentioned that this is particularly useful for labeled
data, we did not really elaborate on what that means. In order to
change that, we will spend the next section solely on working
with data containers that have **targets**. This will put us into
the realm of supervised learning. We will see how we can work
with labeled data containers and what special functionality is
available for them.

.. toctree::
   :maxdepth: 3

   documentation/targets

Now that we have covered all the basics, we can start to discuss
some of the more advanced topics. A particularly important aspect
of modern Machine Learning is what is known as *model selection*.
Most of the time, this boils down to choosing appropriate
hyper-parameters for the model one is working with. To avoid
subtle problems in this selection process, and to reduce variance
of the performance estimates, it is quite common to employ some
kind of **repartitioning strategy** on the training data. Of
course, the partitioning itself is just one part of such a model
selection process, since we still have to somehow compute and
compare the performance. However, it is an important step that is
needed to make the most of the available data. So important in
fact, that we will spend a whole section on it.

.. toctree::
   :maxdepth: 2

   documentation/folds

A different kind of partitioning-need arises from the fact that
the interesting data sets are increasing in size as the
scientific community continues to improve the state-of-the-art in
Machine Learning. While "too much" data is a nice problem to
have, bigger data sets also pose additional challenges in terms
of computing resources. Luckily, there are popular techniques in
place to deal with such constraints in a surprisingly effective
manner. For example, there are a lot of empirical results that
demonstrate the efficiency of optimization techniques that
continuously update on small subsets of the data. As such, it has
become a de facto standard for many algorithms to iterate over a
given dataset in mini-batches, or even just one observation at a
time.

The way this package approaches the topic of data iteration is
complex enough that it deserves two parts. In the first part we
will introduce a few special data iterators, which we call **data
views**, that will allow us to perform such iteration-pattern
conveniently for data containers. In fact, they are more than
"just" data iterators; they are proper vectors. As such, they
also serve as a tool to "view" a data container from a specific
aspect: As a vector of observations, or a vector of batches.
Thus these views know how many observations they contain, and how
to query specific parts of the data.

Furthermore, we also provide a convenient way to partition a long
sequence of data (e.g. some text or time-series) into a vector
of smaller sequences. This is done using a sliding window
approach that supports custom strides and even a self-labeling
function. What that means, and how that works will be discussed
in the next section.

.. toctree::
   :maxdepth: 2

   documentation/dataview

While these data views are also data iterators, the inverse is
not true. In the following section we will introduce a number of
**data iterators**, that don't make *any* other promises than,
well, iteration. As such, they may not know how many observations
they can provide, nor have the means to access specific
observations. Consequently, these data iterators are not data
containers. We will see how that is useful, and also how some of
them are actually created using a data container as input.

.. toctree::
   :maxdepth: 2

   documentation/dataiterator

Indices and tables
==================

.. toctree::
   :hidden:
   :maxdepth: 2

   about/license

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
