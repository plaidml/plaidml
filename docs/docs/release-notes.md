---
nav_order: 7
---

# Release Notes

## PlaidML 0.6.2

* Well defined exports for easier inclusion in other projects & frameworks,
e.g., nGraph
* Initial AMD\* stripe config
* Initial stripe CPU support
* LLVM\* support in windows
* Prototype pytorch\* JIT bridge (limited by pytorch JIT interface)
* Initial C++\* EDSL API support (major revisions expected)

# Previous releases


## PlaidML 0.5.0
  * Support Keras\* 2.2.4
  * Several fixes to Metal\* backend
  * Preliminary release of Stripe
    * New polyhedral IR designed to support modern accelerators
    * Specification, documentation, and paper in progress
    * GPU / OpenCL backend and tutorial coming soon
  * nGraph support (wheels coming soon)
    * Supports tensorflow\* via tensorflow nGraph bridge.


## PlaidML 0.3.3 - 0.3.5
  * Support Keras 2.2.0 - 2.2.2
  * Support ONNX\* 1.2.1
  * Upgrade kernel scheduling
  * Revise documentation
  * Add HALs for CUDA\* and Metal
  * Various bugfixes and improvements


## PlaidML 0.3.2
  * Now supports ONNX 1.1.0 as a backend through
  [onnx-plaidml](https://github.com/plaidml/onnx-plaidml)
  * Preliminary support for LLVM. Currently only supports CPUs, and only on
  Linux and macOS. More soon.
  * Support for LSTMs & RNNs with static loop sizes, such as
  examples/imdb_lstm.py (from Keras)
    * Training networks with embeddings is especially slow (#96)
    * RNNs are only staticly sized if the input's sequence length is explicitly
    specified (#97)
    * Fixes bug related to embeddings (#92)
  * Adds a shared generic op library in python to make creating frontends easier
     * plaidml-keras now uses this library
  * Uses [plaidml/toolchain](https://github.com/plaidml/toolchain) for builds
     * Building for ARM\* is now simple (–-config=linux_arm_32v7)
  * Various fixes for bugs (#89)
