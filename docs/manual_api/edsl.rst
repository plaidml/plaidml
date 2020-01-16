====
EDSL
====

Embedded Domain Specific Language

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::edsl::init

   .. group-tab:: Python

      .. note::

         Initialization of PlaidML's EDSL Python API happens automatically
         wherever the module ``plaidml.edsl`` is imported.

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
      .. autoclass:: plaidml.edsl.Tensor
         :members:
      .. autoclass:: plaidml.edsl.TensorRef
         :members:
      .. autoclass:: plaidml.edsl.ProgramArgument
         :members:
      .. autoclass:: plaidml.edsl.Program
         :members:
      .. autoclass:: plaidml.edsl.TensorDim
         :members:
      .. autoclass:: plaidml.edsl.TensorIndex
         :members:
      .. autoclass:: plaidml.edsl.Constraint
         :members:

----------
Primitives
----------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: edsl_primitives
         :content-only:

   .. group-tab:: Python

      .. autofunction:: plaidml.edsl.abs
      .. autofunction:: plaidml.edsl.cast
      .. autofunction:: plaidml.edsl.cos
      .. autofunction:: plaidml.edsl.cosh
      .. autofunction:: plaidml.edsl.exp
      .. autofunction:: plaidml.edsl.gather
      .. autofunction:: plaidml.edsl.ident
      .. autofunction:: plaidml.edsl.index
      .. autofunction:: plaidml.edsl.log
      .. autofunction:: plaidml.edsl.pow
      .. autofunction:: plaidml.edsl.prng
      .. autofunction:: plaidml.edsl.reshape
      .. autofunction:: plaidml.edsl.scatter
      .. autofunction:: plaidml.edsl.select
      .. autofunction:: plaidml.edsl.shape
      .. autofunction:: plaidml.edsl.sin
      .. autofunction:: plaidml.edsl.sinh
      .. autofunction:: plaidml.edsl.sqrt
      .. autofunction:: plaidml.edsl.tan
      .. autofunction:: plaidml.edsl.tanh

--------
Examples
--------

.. code-block:: c++

   Tensor sum_over_axis(const Tensor& I) {
      TensorDim M, N;
      TensorIndex m, n;
      I.bind_dims(M, N);
      auto O = TensorOutput(N);
      O(n) += I(m, n); // contraction
      return O;
   }

.. math::
   \color{red}O[n]
   \color{default}=
   \color{green}\sum_{m}
   \color{blue}I[m, n]

.. math::
   \color{red}\verb|O(n)|
   \color{green}\verb| += |
   \color{blue}\verb|I(m, n)|\color{default}\verb|;|
