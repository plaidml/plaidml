====
EDSL
====

Embedded Domain Specific Language

.. contents::

.. doxygenfunction:: plaidml::edsl::init

-------
Objects
-------

.. doxygengroup:: edsl_objects
   :content-only:
   :members:

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
