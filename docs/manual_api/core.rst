====
Core
====

PlaidML Core

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::init

   .. group-tab:: Python

      .. autofunction:: plaidml.core.__init

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: core_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. autoclass:: plaidml.core.DType
         :members:
      .. autoclass:: plaidml.core.TensorShape
         :members:
      .. autoclass:: plaidml.core._View
         :members:
      .. autoclass:: plaidml.core.Buffer
         :members:
