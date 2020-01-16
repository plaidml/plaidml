=========
Execution
=========

PlaidML Execution

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::exec::init

   .. group-tab:: Python

      .. note::

         Initialization in Python follows standard conventions: the PlaidML
         Execution APIs are initialized automatically wherever the namespace
         `plaidml.exec` is imported.

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: exec_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. autoclass:: plaidml.exec.Binding
         :members:
      .. autoclass:: plaidml.exec.Executable
         :members:
      .. autoclass:: plaidml.exec.Binder
         :members:
