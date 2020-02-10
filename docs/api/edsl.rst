====
EDSL
====

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::edsl::init

   .. group-tab:: Python

      .. note::

         Initialization of the PlaidML EDSL API occurs when the ``plaidml.edsl``
         module is imported.

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: edsl_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. autoclass:: plaidml.edsl.IndexedTensor
         :members:
         :special-members:
      .. autoclass:: plaidml.edsl.LogicalShape
         :members:
      .. autoclass:: plaidml.edsl.TensorDim
         :members:
         :special-members:
      .. autoclass:: plaidml.edsl.TensorIndex
         :members:
         :special-members:
      .. autoclass:: plaidml.edsl.Tensor
         :members:
         :special-members:
      .. autoclass:: plaidml.edsl.TensorRef
         :members:
      .. autoclass:: plaidml.edsl.ProgramArgument
         :members:
      .. autoclass:: plaidml.edsl.Program
         :members:
      .. autoclass:: plaidml.edsl.Value
         :members:

----------
Primitives
----------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: edsl_primitives
         :content-only:

   .. group-tab:: Python

      .. autofunction:: plaidml.edsl.Placeholder
      .. autofunction:: plaidml.edsl.TensorDims
      .. autofunction:: plaidml.edsl.TensorIndexes
      .. autofunction:: plaidml.edsl.TensorOutput
      .. autofunction:: plaidml.edsl.abs
      .. autofunction:: plaidml.edsl.cast
      .. autofunction:: plaidml.edsl.ceil
      .. autofunction:: plaidml.edsl.cos
      .. autofunction:: plaidml.edsl.cosh
      .. autofunction:: plaidml.edsl.exp
      .. autofunction:: plaidml.edsl.floor
      .. autofunction:: plaidml.edsl.gather
      .. autofunction:: plaidml.edsl.ident
      .. autofunction:: plaidml.edsl.index
      .. autofunction:: plaidml.edsl.log
      .. autofunction:: plaidml.edsl.pow
      .. autofunction:: plaidml.edsl.prng
      .. autofunction:: plaidml.edsl.reshape
      .. autofunction:: plaidml.edsl.round
      .. autofunction:: plaidml.edsl.scatter
      .. autofunction:: plaidml.edsl.select
      .. autofunction:: plaidml.edsl.shape
      .. autofunction:: plaidml.edsl.sin
      .. autofunction:: plaidml.edsl.sinh
      .. autofunction:: plaidml.edsl.sqrt
      .. autofunction:: plaidml.edsl.tan
      .. autofunction:: plaidml.edsl.tanh
