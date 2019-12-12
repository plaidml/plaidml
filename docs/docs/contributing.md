# Contributing to PlaidML

We welcome contributions to PlaidML from anyone. This document contains:

* Guidelines for creating successful PRs
* Outlines the contribution process
* Lists general areas for contribution
* Provides resources and context to ease development, where relevant and
  available

Before starting any work, please ensure you are able to build and test PlaidML.


## Guidelines

* Create unit tests for new features and bug fixes. Integration tests are
  required for larger features.

* Pre-commit linters will be available soon.

  * C++ code conforms to the [Google Style Guide for CPP].
  * Python code conforms to the [Google Python Style Guide].

## Process

1. Ensure there is an open issue assigned to you before doing (too much) work:
   * If you're tackling an open issue that isn't assigned, please assign it to
     yourself.
   * If you'd like to solve an issue that is already assigned, please comment
     on the issue to ensure you're not duplicating effort.
   * If you'd like to provide a new feature, open a new issue. Please provide a
     reasonably-detailed description of what you'd like to do, and clearly
     indicate that you're willing to do the work.
1. Work on a fork as usual in GitHub. Please ensure the same tests travis runs
   will pass before creating a PR.
1. Review the [License] file in the ``plaidml`` repo and the Guidelines on this
page.
1. Once tests have passed, a maintainer will assign the issue to themselves and
   run the PR through the (currently private) performance test suite. If there
   are issues, we will attempt to resolve them, but we may provide details and
   ask the author to address.
1. Once the performance regression suite has passed, we will accept and merge
   the PR.

## Areas for Contribution

* Ops for Frontends
  * PlaidML welcomes implementations for currently unimplemented operations as
    well as Tile code for novel operations supported by research.
  * Please read [adding_ops](adding_ops.md) and
  [writing_tile_code](writing_tile_code.md) tutorials.


* ML Framework Frontends (e.g., Keras, Pytorch, etc)
  * PlaidML welcomes integrations with any established ML framework or interop
    (NNVM, ONNX, etc).
  * You can find commonly used operations in the
    [plaidml.op](api/plaidml.op.rst) module.
  * Please read [building a frontend](building_a_frontend.md) tutorial.


* HALs for Backend Targets (OpenCL, Vulkan, SPIR-V, HVX, etc)
  * There is no documentation for the HAL currently. The interface is fairly
    straightforward and the `OpenCL HAL <../tile/hal/opencl>` provides a good
    example of a complete HAL.

Please follow the process above before embarking on anything major (like
integrating a new frontend or backend).


[Google Style Guide for CPP]: https://google.github.io/styleguide/cppguide.html
[Google Python Style Guide]: https://google.github.io/styleguide/pyguide.html
[License]: https://raw.githubusercontent.com/plaidml/plaidml/master/LICENSE
