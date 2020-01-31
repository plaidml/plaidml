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

      .. note::

         Initialization of PlaidML's Core Python API happens automatically
         wherever the module ``plaidml.core`` is imported.

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
      .. autoclass:: plaidml.core.View
         :members:
      .. autoclass:: plaidml.core.Buffer
         :members:
