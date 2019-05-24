.. contributing.rst: 


Contributing to PlaidML
=======================

We welcome contributions to PlaidML from anyone. This document contains:

* Guidelines for creating successful PRs
* Outlines the contribution process
* Lists general areas for contribution
* Provides resources and context to ease development, where relevant and
  available

Before starting any work, please ensure you are able to build and test PlaidML.


Guidelines
----------

* Create unit tests for new features and bug fixes. Integration tests are
  required for larger features.

* Pre-commit linters will be available soon.

  * C++ code conforms to the `Google Style Guide for CPP`_.
  * Python code conforms to the `Google Python Style Guide`_.

Process
-------

#. Ensure there is an open issue assigned to you before doing (too much) work:

   * If you're tackling an open issue that isn't assigned, please assign it to 
     yourself.
   * If you'd like to solve an issue that is already assigned, please comment 
     on the issue to ensure you're not duplicating effort.
   * If you'd like to provide a new feature, open a new issue. Please provide a
     reasonably-detailed description of what you'd like to do, and clearly 
     indicate that you're willing to do the work.

#. Work on a fork as usual in GitHub. Please ensure the same tests travis runs 
   will pass before creating a PR.
#. Review the `License <https://raw.githubusercontent.com/plaidml/plaidml/master/LICENSE>`_ 
   file in the ``plaidml`` repo and the Guidelines on this page. 
#. Once tests have passed, a maintainer will assign the issue to themselves and 
   run the PR through the (currently private) performance test suite. If there 
   are issues, we will attempt to resolve them, but we may provide details and 
   ask the author to address.
#. Once the performance regression suite has passed, we will accept and merge 
   the PR.



Areas for Contribution
----------------------

* Ops for Frontends

  * PlaidML welcomes implementations for currently unimplemented operations as 
    well as Tile code for novel operations supported by research.
  * Please read the :doc:`adding_ops` and :doc:`writing_tile_code` tutorials.

* ML Framework Frontends (e.g., Keras, Pytorch, etc)

  * PlaidML welcomes integrations with any established ML framework or interop 
    (NNVM, ONNX, etc).
  * You can find commonly used operations in the :doc:`api/plaidml.op` module.
  * Please read the :doc:`building_a_frontend` tutorial.

* HALs for Backend Targets (OpenCL, Vulkan, SPIR-V, HVX, etc)

  * There is no documentation for the HAL currently. The interface is fairly 
    straightforward and the `OpenCL HAL <../tile/hal/opencl>`_ provides a good 
    example of a complete HAL.

Please follow the process above before embarking on anything major (like 
integrating a new frontend or backend).


.. _Google Style Guide for CPP: https://google.github.io/styleguide/cppguide.html 
.. _Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
