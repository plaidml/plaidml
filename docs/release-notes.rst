.. release-notes:

Release Notes
#############

The current release is:  |release|







Previous releases 
=================

* PlaidML 0.3.3 - 0.3.5
  - Support Keras 2.2.0 - 2.2.2
  - Support ONNX 1.2.1
  - Upgrade kernel scheduling
  - Revise documentation
  - Add HALs for CUDA and Metal
  - Various bugfixes and improvements

* PlaidML 0.3.2
  - Now supports ONNX 1.1.0 as a backend through [onnx-plaidml](https://github.com/plaidml/onnx-plaidml)
  - Preliminary support for LLVM. Currently only supports CPUs, and only on Linux and macOS. More soon.
  - Support for LSTMs & RNNs with static loop sizes, such as examples/imdb_lstm.py (from Keras)
    - Training networks with embeddings is especially slow (#96)
    - RNNs are only staticly sized if the input's sequence length is explicitly specified (#97)
    - Fixes bug related to embeddings (#92)
  * Adds a shared generic op library in python to make creating frontends easier
     - plaidml-keras now uses this library
  * Uses [plaidml/toolchain](https://github.com/plaidml/toolchain) for builds
     - Building for ARM is now simple (–-config=linux_arm_32v7)
  * Various fixes for bugs (#89)




.. For example: See also our recent `API changes`_


