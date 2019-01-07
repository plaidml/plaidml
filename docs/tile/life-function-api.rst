.. tile/life-function-api.rst:

=======================
Through the PlaidML API
=======================

The :doc:`PlaidML API <../plaidml>` provides an interface for a frontend like 
Keras or ONNX to request the construction and execution of Tile functions. 
The PlaidML API is used both for generating Tile code and also to request 
compilation and execution of that Tile code; the later also requires 
information on where the input and output data is to be located and on the 
hardware to be used.

Tile Code
---------

This document assumes you already have Tile code. For discussion of how to 
generate Tile code, see :doc:`../building_a_frontend` and :doc:`../writing_tile_code`.

Example
-------

Let's begin by creating a PlaidML function from source code. We will consider 
the case of categorical crossentropy between ``T`` and ``O`` as specified 
by the following Tile code:

.. code-block:: none

    function (T[X, Y], O[X, Y]) -> (R) {
        LO = log(O);
        Temp[x: X] = +(LO[x, y] * T[x, y]);
        R = -Temp;
    }

Typically, we call PlaidML code through a frontend like Keras or ONNX, but we 
can also call the PlaidML API functions directly. We'll do so here to serve as 
an illustration of how the API works, wrapping the Tile code with Python code 
that defines some fake data and which tells PlaidML necessary information about 
the data and hardware:

.. code-block:: python

    import numpy as np
    import plaidml
    
    # Generate input data
    dimensions = (10, 26)
    T = np.zeros(dimensions)
    for i in range(dimensions[0]):
        T[i, 0] = 1.
    O = (np.random.rand(*dimensions) + 1.) / 52.
    O_row_sum = O.sum(axis=1)
    for i in range(dimensions[0]):
        O[i, 0] += 1. - O_row_sum[i]
    
    # Process data using PlaidML
    ctx = plaidml.Context()
    func = plaidml.Function("""function (T[X, Y], O[X, Y]) -> (R) {
            LO = log(O);
            Temp[x: X] = +(LO[x, y] * T[x, y]);
            R = -Temp;
        }""")
    with plaidml.open_first_device(ctx) as dev:
        dtype = plaidml.DType.FLOAT32
        in_shape = plaidml.Shape(ctx, dtype, *dimensions)
        t = plaidml.Tensor(dev, in_shape)
        with t.mmap_discard(ctx) as view:
            view[:] = T.flatten()
            view.writeback()
        o = plaidml.Tensor(dev, in_shape)
        with o.mmap_discard(ctx) as view:
            view[:] = O.flatten()
            view.writeback()
        out_shape = plaidml.Shape(ctx, dtype, *dimensions[:-1])
        r = plaidml.Tensor(dev, out_shape)
        plaidml.run(ctx, func, inputs={"T": t, "O": o}, outputs={"R": r})
        with r.mmap_current() as view:
            R = view[:]
    print(R)    # Report results


Tensor Metadata
---------------

The execution model is based on the idea of a data dependency graph. In this 
trivial example, our graph is correspondingly trivial, but we can see all of 
the components in action.

A :doc:`../api/plaidml.Tensor` represents a multidimensional array, combining a 
memory buffer and a :doc:`../api/plaidml.tile.Shape` describing the number and 
extent of its dimensions. The contents of the tensor will be moved wherever 
they are needed, so the buffer may be located in system memory or on the GPU.

In order to read or write the contents of a tensor, you must mmap it into 
system memory. You can ``mmap_discard`` if you don't care about the existing 
contents of the buffer, and simply want a writable view you can populate, or 
you can ``mmap_current`` to preserve the contents of the tensor, as would be 
appropriate when you want to use the function's output. In either case, the 
mmap function provides a ``plaidml._View`` object.

When populating our input tensors, ``t`` and ``o``, we use ``mmap_discard`` 
because these freshly-allocated buffers have no data. After we've finished 
writing values into the view, we indicate that the modified data is complete 
by calling ``writeback``.

The ``view`` object serves as a lock, and it is this lock which allows 
asynchronous, possibly multithreaded function execution to be synchronized. 
When a function runs, it must first mmap in its input and output buffers, 
and the mmap operation will block until the current view has been released. 
In this example we are using the ``view`` as a context manager, to ensure 
that it will be closed after we are done populating each tensor.


Execution
---------

The call to ``plaidml.run`` begins by creating a :doc:`../api/plaidml.Function` 
for the given source code. Next, it creates a :doc:`../api/plaidml.Invoker` 
which will execute that function in the given context. Each input and output 
tensor is bound to the :doc:`../api/plaidml.Invoker` instance. Finally, 
``plaidml.run`` calls the invoker's ``invoke()`` method, returning a 
:doc:`../api/plaidml.Invocation`, and this invocation is what actually schedules 
execution of the bound function.

Inside the PlaidML runtime, the scheduler passes the :doc:`../api/plaidml.Invocation` 
instance off to a hardware support module, which compiles the Tile function 
appropriately for the target device. The PlaidML runtime then launches the 
executable code and returns, leaving the function to run asynchronously whenever 
its dependencies become available.


Composition
-----------

This process works reasonably well provided you can write your entire program 
as a single Tile function. It is possible to read out the result of one Tile 
function and pass it to another via the mmap operations, but this will hurt 
performance. The best way to compose multiple Tile functions is by using the 
:doc:`../api/plaidml.tile.Operation` and :doc:`../api/plaidml.tile.Value` objects 
provided in :doc:`../api/plaidml.tile`; see :doc:`../building_a_frontend` for details. 
This results in an :doc:`../api/plaidml.Invocation` and output(s) that can be read 
via ``mmap_current`` in the same way as a single Tile function.

