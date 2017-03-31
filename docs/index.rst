.. MLDataPattern.jl documentation master file, created by
   sphinx-quickstart on Fri Mar 31 14:06:42 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MLDataPattern.jl's documentation!
=====================================

This package represents a community effort to provide a native
Julia implementation for common data access pattern in Machine
Learning. This includes widely used access pattern for shuffling,
partitioning, and resampling data sets of various types. More
importantly, the package was designed around the core premise of
allowing any user-defined type to serve as custom data sources
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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

