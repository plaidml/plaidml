.. tile/life-function-compiler.rst:

=========================
Through the Tile Compiler
=========================

The Tile compiler takes Tile code and produces symbolic descriptions of the 
hardware kernels needed to execute it. It sits between the PlaidML API, which 
interacts with a frontend to produce Tile code, and the Hardware Abstraction 
Layer, which converts kernel descriptions into compiled kernels for the 
available hardware.

Inputs
------

When the PlaidML API invokes the Tile compiler, the function arrives in the 
form of a ``tile::proto::Program``. This ``Program`` protobuf combines the 
function's source code, a target device, and a description of the input and 
output tensors; it is analogous to a :doc:`../api/plaidml.Invoker`.

The PlaidML API also creates an ``Evaluator`` object for each device; the 
evaluator contains device settings and a ``ProgramCache``. The ``ProgramCache`` 
determines for each given function whether a previously compiled executable 
exists or whether it needs to be compiled, then stores the compiled representation 
for later use.

The first time a function is executed, then, we must compile it. We begin by 
associating the ``tile::proto::Program`` with a specific device, yielding a 
``tile::local_machine::Program``. This adds parameters about device 
characteristics which the compiler will use for performance estimation, such as 
the number of registers and the maximum vector width, as defined in 
``tile::hal::HardwareSettings``.

Parsing
-------

Compilation begins by parsing the Tile language source code, using a standard 
lex-based lexer and yacc-based parser. Parser output is a ``tile::lang::Program``, 
which is not the same as a ``tile::proto::Program``, nor a 
``tile::local_machine::Program``. In this context, a ``Program`` is essentially 
a list of ``Op`` instances, plus lists of the input & output variables. An 
``Op`` is essentially a line of Tile code in semi-symbolic form. While this 
``Program`` functions at a level of abstraction similar to an AST, it is a list 
and not a tree. After parsing, the function's representation as a ``Program`` 
looks like this:

.. code-block:: none

   function (
     X_I_0[X_I_0_0, X_I_0_1],
     X_I_1[X_I_1_0, X_I_1_1]
   ) -> (
     X_T3
   ) {
     X_T2 = log(X_I_0);
     _T1 = 3;
     X_T0[x : _T1] = +(X_T2[x, y] * X_I_1[x, y]);
     X_T3 = neg(X_T0);
   }

The first two indented lines define the inputs, ``X_I_0`` and ``X_I_1``, with 
their index variables, and the second indented block defines the output, ``X_T3``. 
The body of the function in the last indented block is its sequence of ``Op`` 
instances. With one output and zero or more inputs, ``Op``\s can be:

* Constants, which have some fixed value
* Functions, applied to each element of a tensor
* Contractions, combining elements from two input tensors

GenerateProgram
---------------

After Tile code has been parsed into a list of ops, ``lang::GenerateProgram`` 
is used to construct ``lang::KernelInfo`` objects, which are symbolic descriptions 
of kernels. There is not an exact correspondence between ops and kernels 
(constants may be inlined, elementwise functions can be merged, etc), but 
especially for complex operations the produced kernels and the operations that
generate them do roughly correspond.

Compiling Contractions
~~~~~~~~~~~~~~~~~~~~~~

(See :doc:`../writing_tile_code` for more details about contraction semantics.)

One of the most involved compilation steps is creating kernels for contractions. 
A contraction traverses tensors using a multidimensional strided access pattern. 
It can traverse just one tensor, in which case it merely accesses values, or it 
can traverse two tensors, in which case it accesses pairs of values and combines 
them using a simple operation, such as addition or multiplication. If the 
contraction produces multiple values for a single output cell, it will aggregate 
the output values using some other simple reduction operation (such as sum or max). 
Finally, it writes the result to the output tensor.

``lang::Compile`` analyzes the contractions and converts them into 
``FlatContraction``\s. The ``FlatContraction`` is a simplification of the generic 
``Contraction`` access pattern, making assumptions which allow efficient code 
generation. The conversion process applies a series of transforms, including:

* Iterating over fixed bounds for index variables, rather than using the 
  representation "any integers which would be in bounds for all formulas"
* Turning indices into single variables, rather than formulas, whenever possible
* Reducing the number of multiple-index comparisons (expressions like ``i + 2*k < 9``)
* Disentangling output and input indices, allowing iteration over output indices in outer loops such that each index produces a single output

Optimization
~~~~~~~~~~~~

These simplifications make it possible to automatically generate specifications 
for a kernel for a ``FlatContraction``. This includes using ``lang::TileOptimizer`` 
to simulate the performance of the kernel on the target device with varying tile 
sizes, thus enabling flat contractions to be automatically converted into 
performant kernels.

``GenerateProgram`` also processes constants and elementwise functions, as well 
as special operations (e.g. psuedorandom number generation). It may produce 
additional kernels not explicitly specified in Tile (e.g. initializing tensors 
to zero). It looks for opportunities to reduce the number of kernels (e.g. if 
you are taking the log of the absolute value of a tensor those operations can 
generally happen in the same kernel).

``GenerateProgram`` produces a ``lang::KernelInfo`` object for each kernel to be 
constructed. A ``KernelInfo`` describes the inputs and outputs for a single kernel, 
along with some execution shape parameters. It represents the body of the kernel 
not as a list of operations, but as a semantic tree, or "semtree".

A semtree is an intermediate representation which describes each function in 
terms of its executable semantics. Its level of abstraction sits midway between 
an AST and the low-level LLVM IR. While the semtree representation is 
platform-independent, each semtree instance is device-specific, because 
``lang::GenerateProgram`` generates code which is optimized for a specific 
piece of hardware.

The produced ``KernelInfo``\s are rewritten via ``lang::Simplify`` to produce 
equivalent output via more efficient code. These are then combined with tensor 
type information describing the buffers which will be passed between kernels; 
together they form a ``lang::KernelList``.

Compiler Output
---------------

The Tile compiler outputs semtrees encoded as a ``lang::KernelList``. Code 
generation and execution are provided by the hardware support module.
