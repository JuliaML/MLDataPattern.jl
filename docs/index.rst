.. MLDataPattern.jl documentation master file, created by
   sphinx-quickstart on Fri Mar 31 14:06:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLDataPattern.jl's documentation
=====================================

This package represents a community effort to provide a native
and generic `Julia <http://julialang.org>`_ implementation for
common data access pattern in Machine Learning. This includes
widely used access pattern for shuffling, partitioning, and
resampling data sets of various types and origin. More
importantly, the package was designed around the core premise of
allowing any user-defined type to serve as a custom data source
and/or access pattern in a first class manner.

MLDataPattern is a part of the `JuliaML
<https://github.com/JuliaML>`_ ecosystem. In contrast to other
data-centered packages, it focuses specifically on functionality
utilized in a Machine Learning context.

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
many observations they contain as well as how to access specific
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
   :maxdepth: 2

   documentation/datasubset

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

