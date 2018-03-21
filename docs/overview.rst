PlaidML Architecture Overview
=============================
At a High Level PlaidML Consists of:

- A core that exposes a C and C++ API:

  - A HAL API and a library of backends that implement it (OpenCL/LLVM/etc)

  - A runtime which takes tile code, optimizes it based on parameters from the HAL, and a Platform that schedules 
    operations and memory layout based on the type of Platform (Local / Remote)

- Python bindings built on top of the C API

  - An operations library which is a generic library of tile code

  - An API that can be called directly or used to develop other frontends

- Frontend adapters that utilize the op library and the API to implement support for that frontend

  - ONNX

  - Keras

.. uml:: architecture.puml

  