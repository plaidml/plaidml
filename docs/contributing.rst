Contributing to PlaidML
=======================

We welcome contributions to PlaidML from anyone. This document contains:


* Guidelines for creating successful PRs

* Outlines the contribution process

* Lists general areas for contribution

* Provides resources and context to ease development, where relevant and
  available

Before starting any work please ensure you are able to build and test PlaidML.


Guidelines
----------

* Create unit tests for new features and bug fixes. Integration tests are
  required for larger features.

* Pre-commit linters will be available soon.

  * C++ code conforms to the `Google C++ Style Guide`_.

  * Python code conforms to the `Google Python Style Guide`_.

.. _`Google C++ Style Guide`: https://google.github.io/styleguide/cppguide.html
.. _`Google Python Style Guide`: https://google.github.io/styleguide/pyguide.html


Process
-------

#. Ensure there is an open issue assigned to you before doing (too much) work:

   * If you're tackling an open issue that isn't assigned, please assign it to yourself.
   * If you'd like to solve an issue that is already assigned, please comment on the issue
     to ensure you're not duplicating effort.
   * If you'd like to provide a new feature, open a new issue. Please provide a
     reasonably detailed description of what you'd like to do, and clearly indicate that 
     you're willing to do the work.

#. Work on a fork as usual in Github. Please ensure the same tests travis runs will pass
   before creating a PR
#. Once you've created a PR you'll be asked to review and sign our `Contributor License Agreement <https://cla-assistant.io/plaidml/plaidml>`_. 
   You should review this before doing too much work to ensure the terms are acceptable.
#. Once tests have passed, a maintainer will assign the issue to themselves and run the
   PR through the (currently private) performance test suite. If there are issues, we
   will attempt to resolve them, but we may provide details and ask the author to address.
#. Once the performance regression suite has passed, we will accept and merge the PR.


Areas for Contribution
----------------------

* Ops for Frontends

  * PlaidML welcomes implementations for currently unimplemented operations as well as Tile code
    for novel operations supported by research.
  * Please read the `Tile Tutorial <https://github.com/plaidml/plaidml/wiki/Tile-Tutorial>`_ and the `PlaidML Op Tutorial <https://github.com/plaidml/plaidml/wiki/PlaidML-Op-Tutorial>`_ 

* ML Framework Frontends (e.g., Keras, Pytorch, etc)

  * PlaidML welcomes integrations with any established ML framework or interop (NNVM, ONNX, etc)
  * Currently this involves duplicating tile operations. We will eventually abstract common NN tile operations
    into a separate C++ library to ease backend development.

* HALs for Backend Targets (OpenCL, Vulkan, SPIR-V, HVX, etc)

  * There is no documentation for the HAL currently. The interface is fairly straightforward and the `OpenCL HAL <tile/hal/opencl>`_ 
    provides a good example of a complete HAL.

Please follow the process above before embarking on anything major (like integrating a new frontend or backend).

