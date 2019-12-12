# Building a Frontend

PlaidML currently supports Keras and ONNX as frontends. This document explains
how to add PlaidML as a backend to other machine learning frontends.

## Read the Frontend's Documentation

Most frontends document their backend API. Frontends differ in their
requirements of backends, and you may need to adapt these instructions to the
needs of the frontend. Reading the frontend documentation will help you
implement the necessary features in the expected way.

You may also find unit tests available for running the frontend using PlaidML.

## API Overview

The modules [plaidml](plaidml), [api/plaidml.op](api/plaidml.op.rst), and
[api/plaidml.tile](api/plaidml.tile.rst) are of particular use in constructing a
frontend. They respectively provide general purpose classes, a suite of commonly
used operations, and a library for constructing and composing operations.

## Required Functionality

Broadly speaking, to implement a frontend you will need to provide the
following:
 * A means of communicating with the execution device, as discussed in
 [context & device](#context-&-device)
 * Implementations of operations requested by the frontend, as discussed in
 [individual operations](#individual-operations)
 * Construction of a computation graph, as discussed in [functions](#functions)
 and [operations values](operations-values). The frontend may handle some of
 this.
 * A supply of input data to a computation graph, execution of that graph, and
 recovery of its output, as discussed in [functions](#functions) and
 [invokers](#invokers)

The frontend's documentation may describe additional functionality that is also
required.

## Context & Device

The class [api/plaidml.context.Context] provides a context for executing code on
a device as requested by PlaidML via an invoker. Interfacing with a frontend
thus requires a [api/plaidml.context.Context] that is linked to a
[api/plaidml.Device] and [api/plaidml.Invoker](api/plaidml.Invoker.rst)s. A
frontend implementation will need to construct instances of all these classes.

A frontend typically uses a single [api/plaidml.context.Context] and
[api/plaidml.Device]; the [api/plaidml.context.Context] can be constructed
directly e.g. ``_ctx = plaidml.Context()``, while the [api/plaidml.Device] needs
to be initialized with a [api/plaidml.context.Context] and configuration
settings (see ``_device()`` in the Keras frontend or ``_get_device_configs`` and
``PlaidMLBackend.prepare`` in the ONNX frontend for examples).

## Functions

The [api/plaidml.Function] object holds the computation graph of a network.
[api/plaidml.Function]s are typically constructed using
[api/plaidml.tile.compose](api/plaidml.tile.compose.rst). This requires lists of
the initial input and final output variables.

The inputs are provided as pairs of variable names and placeholders. The
placeholders do not yet include the input data but do tell PlaidML what format
of input data to expect. At a minimum, the number of dimensions of each input
must be provided; if this is all you have, inputs can be constructed with
[api/plaidml.tile.Value](api/plaidml.tile.Value.rst).from_ndims. If you know the
size of one or more of the dimensions, it is better to provide the shape by
constructing with
[api/plaidml.tile.Value](api/plaidml.tile.Value.rst).from_dimensions. You may
also need to provide type information (see [dtypes](#dtypes)).

Weights may be included as inputs in addition to the main input data, if the
weights are expected to change between runs (e.g. in training). Data that will
be constant between runs should be used in constructing the output variables
instead.

The outputs are provided as pairs of variable names and operation output
[api/plaidml.tile.Value](api/plaidml.tile.Value.rst)s (returned from an
[api/plaidml.tile.Operation](api/plaidml.tile.Operation.rst) via
``sole_output()``, ``outputs``, or ``output_tuple``). Only the output variables
returned to the user are provided here; intermediate outputs are used only in
the construction of
[api/plaidml.tile.Operation](api/plaidml.tile.Operation.rst)s, as discussed in
[operations values](#operations-values).

Side effects can be also be built into a [api/plaidml.Function] via the ``update`` parameter.

You must also provide a context and device (see
[context & device](context-&-device)).

## Invokers

[api/plaidml.Invoker](api/plaidml.Invoker.rst)s are used to execute
[api/plaidml.Function]s. An [api/plaidml.Invoker](api/plaidml.Invoker.rst) is
constructed from a [api/plaidml.Function] and a [api/plaidml.context.Context],
and must be provided with concrete input data to fill in the input variable
placeholders in the [api/plaidml.Function] and with output variables which
will receive the output data produced by the [api/plaidml.Function].
Different input and output variables may be used each time the
[api/plaidml.Invoker](api/plaidml.Invoker.rst) is invoked.

## Operations & Values

Implementing [api/plaidml.tile.Operation]s for a frontend involves two
broad tasks:
1. Providing an implementation of each type of operation the frontend
may request, which is discussed in
[individual_operations](#individual-operations), and
1. Connecting the [api/plaidml.tile.Operation]s into a computation graph, which
we discuss here.

The [api/plaidml.tile.Value] class is used to store and transfer PlaidML data,
and the [api/plaidml.tile.Operation] class is used to manipulate data. These are
connected to form a computation graph of the desired neural network. More
details are available in [api/plaidml.tile](api/plaidml.tile.rst), but for the
purposes of constructing a computation graph each [api/plaidml.tile.Operation]
will need to be provided input data as [api/plaidml.tile.Value]s in one of the
following forms:
 * Input placeholders, constructed and provided to
 [api/plaidml.tile.compose](api/plaidml.tile.compose.rst) as discussed in
 [functions](#functions).
 * Constants, often constructed via [api/plaidml.tile.Value].from_var or
 [api/plaidml.tile.Value].from_python_value, or at a lower level by manipulating
 a [api/plaidml.Tensor](api/plaidml.Tensor.rst). This may use an initialization
 function provided by the frontend.
 * The output of already-constructed [api/plaidml.tile.Operation]s. This can be
 accessed via ``op.sole_output()`` (if the operation has only one output), by
 ``op.outputs[name]`` (where ``name`` is the name of the desired output
 variable), or by ``op.output_tuple[i]`` (where ``i`` is the index of the
 desired output).

This may be handled somewhat automatically by your frontend. For example, the
Keras frontend API requires a few functions to be implemented (i.e.
``placeholder``, ``variable``, ``constant``, ``zeros``, ...) and then uses these
to provide appropriate data when calling operation-constructing functions.

## DTypes

Note that [api/plaidml.tile.Value]s have an associated data type, which
sometimes must be manually specified. The PlaidML datatypes are specified in
[api/plaidml.DType](api/plaidml.DType.rst); you will probably need to create a
correspondence between these and the frontend's data types.

## Individual Operations

PlaidML operations that are common to multiple frontends can be found in the
[api/plaidml.op](api/plaidml.op.rst) module (a few, such as the Python numeric
type operations, instead appear in [api/plaidml.tile](api/plaidml.tile.rst)).
Some operations can be used directly from the common ops library, e.g. for Keras
the function ``tanh`` is defined as

```python
tanh = op.tanh
```

and for ONNX

```onnx
@staticmethod
@opset_op('Tanh')
def tanh(value):
    return (op.tanh(value),)
```

Others might need a thin wrapper to translate the API. For example with Keras,
``sum`` is defined as

```python
def sum(x, axis=None, keepdims=False):
    return op.summation(
        x, axes=axis, keepdims=keepdims, floatx=ptile.NUMPY_DTYPE_TO_PLAIDML[floatx()])
```

and for ONNX

```onnx
@staticmethod
@opset_op('Sum')
def sum(*args):
    return (functools.reduce(lambda x, y: x + y, args),)
```

(Note that the ``+`` in the reduce comes from
[api/plaidml.tile](api/plaidml.tile.rst) as ``Value.__add__``).

Other operations might not exist in the common op library, and will need to be
defined in whole or in part in a frontend-specific op library; e.g. the
operations ``switch`` and ``tile`` for Keras and the operations ``flatten`` and
``split`` for ONNX. If the operation you wish to implement does not yet exist,
you will need to write Tile code for it (see
[writing_tile_code](writing_tile_code.md)) and wrap that code with a PlaidML
[api/plaidml.tile.Operation] (see [adding_ops](adding_ops.md)).

[api/plaidml.context.Context]: api/plaidml.context.Context.rst
[api/plaidml.Device]: api/plaidml.Device.rst
[api/plaidml.Function]: api/plaidml.Function.rst
[api/plaidml.tile.Operation]: api/plaidml.tile.Operation.rst
[api/plaidml.tile.Value]: api/plaidml.tile.Value.rst
