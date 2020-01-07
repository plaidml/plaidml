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

      .. autofunction:: plaidml2.edsl.__init

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: edsl_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. autoclass:: plaidml2.edsl.IndexedTensor
      .. autoclass:: plaidml2.edsl.LogicalShape
         :members:
      .. autoclass:: plaidml2.edsl.Tensor
      .. autoclass:: plaidml2.edsl.TensorRef
      .. autoclass:: plaidml2.edsl.ProgramArgument
      .. autoclass:: plaidml2.edsl.Program
      .. autoclass:: plaidml2.edsl.TensorDim
      .. autoclass:: plaidml2.edsl.TensorIndex
      .. autoclass:: plaidml2.edsl.Constraint

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
