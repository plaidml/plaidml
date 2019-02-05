.. building_a_frontend.rst:

===================
Building a Frontend
===================

PlaidML currently supports Keras and ONNX as frontends. This document explains 
how to add PlaidML as a backend to other machine learning frontends.

.. _frontend_docs:

Read the Frontend's Documentation
---------------------------------
Most frontends document their backend API. Frontends differ in their requirements of backends, and you may need to adapt these instructions to the needs of the frontend. Reading the frontend documentation will help you implement the necessary features in the expected way.

You may also find unit tests available for running the frontend using PlaidML.

.. _api_overview:

API Overview
------------
The modules :doc:`plaidml <plaidml>`, :doc:`api/plaidml.op`, and :doc:`api/plaidml.tile` are of particular use in constructing a frontend. They respectively provide general purpose classes, a suite of commonly used operations, and a library for constructing and composing operations.

.. _required_functionality:

Required Functionality
----------------------
Broadly speaking, to implement a frontend you will need to provide the following:
 * A means of communicating with the execution device, as discussed in :ref:`context_device`
 * Implementations of operations requested by the frontend, as discussed in :ref:`individual_operations`
 * Construction of a computation graph, as discussed in :ref:`functions` and :ref:`operations_values`. The frontend may handle some of this.
 * A supply of input data to a computation graph, execution of that graph, and recovery of its output, as discussed in :ref:`functions` and :ref:`invokers`

The frontend's documentation may describe additional functionality that is also required.

.. _context_device:

Context & Device
================
The class :doc:`api/plaidml.context.Context` provides a context for executing code on a device as requested by PlaidML via an invoker. Interfacing with a frontend thus requires a :doc:`api/plaidml.context.Context` that is linked to a :doc:`api/plaidml.Device` and :doc:`api/plaidml.Invoker`\s. A frontend implementation will need to construct instances of all these classes.

A frontend typically uses a single :doc:`api/plaidml.context.Context` and :doc:`api/plaidml.Device`; the :doc:`api/plaidml.context.Context` can be constructed directly e.g. ``_ctx = plaidml.Context()``, while the :doc:`api/plaidml.Device` needs to be initialized with a :doc:`api/plaidml.context.Context` and configuration settings (see ``_device()`` in the Keras frontend or ``_get_device_configs`` and ``PlaidMLBackend.prepare`` in the ONNX frontend for examples).

.. _functions:

Functions
=========
The :doc:`api/plaidml.Function` object holds the computation graph of a network. :doc:`api/plaidml.Function`\s are typically constructed using :doc:`api/plaidml.tile.compose`. This requires lists of the initial input and final output variables.

The inputs are provided as pairs of variable names and placeholders. The placeholders do not yet include the input data but do tell PlaidML what format of input data to expect. At a minimum, the number of dimensions of each input must be provided; if this is all you have, inputs can be constructed with :doc:`api/plaidml.tile.Value`.from_ndims. If you know the size of one or more of the dimensions, it is better to provide the shape by constructing with :doc:`api/plaidml.tile.Value`.from_dimensions. You may also need to provide type information (see :ref:`dtypes`).

Weights may be included as inputs in addition to the main input data, if the weights are expected to change between runs (e.g. in training). Data that will be constant between runs should be used in constructing the output variables instead.

The outputs are provided as pairs of variable names and operation output :doc:`api/plaidml.tile.Value`\s (returned from an :doc:`api/plaidml.tile.Operation` via ``sole_output()``, ``outputs``, or ``output_tuple``). Only the output variables returned to the user are provided here; intermediate outputs are used only in the construction of :doc:`api/plaidml.tile.Operation`\s, as discussed in :ref:`operations_values`.

Side effects can be also be built into a :doc:`api/plaidml.Function` via the ``update`` parameter.

You must also provide a context and device (see :ref:`context_device`).

.. _invokers:

Invokers
========
:doc:`api/plaidml.Invoker`\s are used to execute :doc:`api/plaidml.Function`\s. An :doc:`api/plaidml.Invoker` is constructed from a :doc:`api/plaidml.Function` and a :doc:`api/plaidml.context.Context`, and must be provided with concrete input data to fill in the input variable placeholders in the :doc:`api/plaidml.Function` and with output variables which will receive the output data produced by the :doc:`api/plaidml.Function`. Different input and output variables may be used each time the :doc:`api/plaidml.Invoker` is invoked.

.. _operations_values:

Operations & Values
===================
Implementing :doc:`api/plaidml.tile.Operation`\s for a frontend involves two broad tasks: providing an implementation of each type of operation the frontend may request, which is discussed in :ref:`individual_operations`, and connecting the :doc:`api/plaidml.tile.Operation`\s into a computation graph, which we discuss here.

The :doc:`api/plaidml.tile.Value` class is used to store and transfer PlaidML data, and the :doc:`api/plaidml.tile.Operation` class is used to manipulate data. These are connected to form a computation graph of the desired neural network. More details are available in :doc:`api/plaidml.tile`, but for the purposes of constructing a computation graph each :doc:`api/plaidml.tile.Operation` will need to be provided input data as :doc:`api/plaidml.tile.Value`\s in one of the following forms:
 * Input placeholders, constructed and provided to :doc:`api/plaidml.tile.compose` as discussed in :ref:`functions`.
 * Constants, often constructed via :doc:`api/plaidml.tile.Value`.from_var or :doc:`api/plaidml.tile.Value`.from_python_value, or at a lower level by manipulating a :doc:`api/plaidml.Tensor`. This may use an initialization function provided by the frontend.
 * The output of already-constructed :doc:`api/plaidml.tile.Operation`\s. This can be accessed via ``op.sole_output()`` (if the operation has only one output), by ``op.outputs[name]`` (where ``name`` is the name of the desired output variable), or by ``op.output_tuple[i]`` (where ``i`` is the index of the desired output).

This may be handled somewhat automatically by your frontend. For example, the Keras frontend API requires a few functions to be implemented (i.e. ``placeholder``, ``variable``, ``constant``, ``zeros``, ...) and then uses these to provide appropriate data when calling operation-constructing functions.

.. _dtypes:

DTypes
------

Note that :doc:`api/plaidml.tile.Value`\s have an associated data type, which sometimes must be manually specified. The PlaidML datatypes are specified in :doc:`api/plaidml.DType`; you will probably need to create a correspondence between these and the frontend's data types.

.. _individual_operations:

Individual Operations
=====================

PlaidML operations that are common to multiple frontends can be found in the 
:doc:`api/plaidml.op` module (a few, such as the Python numeric type operations, 
instead appear in :doc:`api/plaidml.tile`). Some operations can be used directly 
from the common ops library, e.g. for Keras the function ``tanh`` is defined as 

::

  tanh = op.tanh

and for ONNX 

::

  @staticmethod
  @opset_op('Tanh')
  def tanh(value):
      return (op.tanh(value),)

Others might need a thin wrapper to translate the API. For example with Keras, 
``sum`` is defined as 

::

  def sum(x, axis=None, keepdims=False):
      return op.summation(
          x, axes=axis, keepdims=keepdims, floatx=ptile.NUMPY_DTYPE_TO_PLAIDML[floatx()])

and for ONNX 

::

  @staticmethod
  @opset_op('Sum')
  def sum(*args):
      return (functools.reduce(lambda x, y: x + y, args),)

(Note that the ``+`` in the reduce comes from :doc:`api/plaidml.tile` as '
``Value.__add__``).

Other operations might not exist in the common op library, and will need to be 
defined in whole or in part in a frontend-specific op library; e.g. the operations 
``switch`` and ``tile`` for Keras and the operations ``flatten`` and ``split`` 
for ONNX. If the operation you wish to implement does not yet exist, you will 
need to write Tile code for it (see :doc:`writing_tile_code`) and wrap that code 
with a PlaidML :doc:`api/plaidml.tile.Operation` (see :doc:`adding_ops`).
