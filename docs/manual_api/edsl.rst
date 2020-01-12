====
EDSL
====

Embedded Domain Specific Language

.. contents::

-------
Initialization
-------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::edsl::init

   .. group-tab:: Python

      .. autofunction:: plaidml.edsl.__init

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
      .. autoclass:: plaidml.edsl.LogicalShape
         :members:
      .. autoclass:: plaidml.edsl.Tensor
      .. autoclass:: plaidml.edsl.TensorRef
      .. autoclass:: plaidml.edsl.ProgramArgument
      .. autoclass:: plaidml.edsl.Program
      .. autoclass:: plaidml.edsl.TensorDim
      .. autoclass:: plaidml.edsl.TensorIndex
      .. autoclass:: plaidml.edsl.Constraint

----------
Primitives
----------

.. doxygengroup:: edsl_primitives
   :content-only:

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
