=========
Execution
=========

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::exec::init

   .. group-tab:: Python

      .. note::

         Initialization of the PlaidML Execution API occurs when the
         ``plaidml.exec`` module is imported.

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: exec_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. autoclass:: plaidml.exec.Executable
         :members:
      .. autoclass:: plaidml.exec.Binder
         :members:
