PlaidML Architecture Overview
=============================

.. TODO  -- this needs updated with visual illustration and nGraph-specific info.


At a high level PlaidML consists of:

- **A core that exposes a C and C++ API** named ``PlaidML``:

  - A HAL API and a library of backends that implement it (OpenCL/LLVM/etc)

  - A runtime which takes tile code, optimizes it based on parameters from the 
    HAL, and a Platform that schedules operations and memory layout based on the 
    type of platform (for example, local vs remote).

- **Python bindings** built on top of the C API:

  - An operations library which is a generic library of tile code

  - An API that can be called directly or used to develop other frontends

- **Frontend adapters** that utilize the ``op`` library and the API to implement 
  support for that frontend:

  - ONNX

  - Keras

