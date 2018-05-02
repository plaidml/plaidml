=======================
Life of a Tile Function
=======================

PlaidML represents machine learning operations as functions in the Tile programming language. PlaidML compiles functions written in the Tile language into optimized kernels for various backends. This document walks through the process of transforming a Tile function into OpenCL code and machine code via LLVM.

While most people will choose to either use PlaidML as a backend for Keras or ONNX or to build operation graphs using the higher-level :doc:`api/plaidml.op` and :doc:`api/plaidml.tile` APIs, these interfaces are all implemented in terms of Tile functions.

We can therefore take a tour through the core components in PlaidML's architecture by following the course a simple Tile function takes from definition through execution. For this example, we'll use the same ``categorical_crossentropy`` function explored in the :doc:`adding_ops` tutorial:

.. code-block:: none

   function (T[X, Y], O[X, Y]) -> (R) {
      LO = log(O);
      Temp[x: X] = +(LO[x, y] * T[x, y]);
      R = -Temp;
   }


We'll consider how this Tile function passes through three major stages of PlaidML. First, the PlaidML API parses the raw string containing Tile code and the data into an :doc:`api/plaidml.Invoker` or :doc:`api/plaidml.Invocation`, which contains symbolic information detailing what computations to execute. Next, the Tile compiler transforms this :doc:`api/plaidml.Invoker` into a list of semantic trees, each detailing a kernel that needs to be executed. Finally, the hardware abstraction layer transforms these semantic trees into device-specific kernels.

.. toctree::
   :maxdepth: 1

   life-function-api
   life-function-compiler
   life-function-hal



