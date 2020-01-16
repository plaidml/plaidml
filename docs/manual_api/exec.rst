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

         Initialization of PlaidML's Execution Python API happens
         automatically wherever the module ``plaidml.exec`` is imported.

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
