====
Core
====

.. contents::

--------------
Initialization
--------------

.. tabs::

   .. group-tab:: C++

      .. doxygenfunction:: plaidml::init

   .. group-tab:: Python

      .. note::

         Initialization of the PlaidML Core API occurs when the ``plaidml``
         module is imported.

-------
Objects
-------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: core_objects
         :content-only:
         :members:

   .. group-tab:: Python

      .. automodule:: plaidml.core
         :members:

--------
Settings
--------

.. tabs::
   .. group-tab:: C++

      .. doxygengroup:: core_settings
         :content-only:
         :members:

   .. group-tab:: Python

      .. autofunction:: plaidml.settings.all
      .. autofunction:: plaidml.settings.get
      .. autofunction:: plaidml.settings.set
      .. autofunction:: plaidml.settings.load
      .. autofunction:: plaidml.settings.save

