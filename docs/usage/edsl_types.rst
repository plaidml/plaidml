eDSL Types
##########
The following documentation describes the primitive types that exist in the Tile eDSL, and shows several usage examples.

Tensor Types
============

- ``Tensor``: Multidimensional arrays of a fixed shape. The scope of a tensor is
  the entire function. By convention, tensors begin with a capital letter.
- ``TensorDim``: Positive integers initially passed to a function as sizes of
  input tensors. The scope of a dimension is the entire function. By convention,
  dimensions begin with a capital letter.
- ``TensorIndex``: Symbolic integers used in contractions to directly index a
  tensor or as part of a formula to compute a tensor index. The scope of an
  index is a single operation. By convention, indices begin with a lower case
  letter.

In the following example, the ``Tensor`` is shown in blue, the ``TensorDim`` is shown in red, and the ``TensorIndexes`` are shown in green.

.. math::
  \color{default}\verb!Contraction().outShape(!
  \color{red}\verb!N!
  \color{default}\verb!).outAccess(!
  \color{green}\verb!n!
  \color{default}\verb!).sum(!
  \color{blue}\verb!I!
  \color{default}\verb![!
  \color{green}\verb!m, n!
  \color{default}\verb!])!

Data Types
==========
The following data types are defined in the eDSL:

- ``INTX``: signed integer with arbitrary precision
- ``INT8``: signed integer with 8-bit precision
- ``INT16``: signed integer with 16-bit precision
- ``INT32``: signed integer with 32-bit precision
- ``INT64``: signed integer with 64-bit precision
- ``UINTX``: unsigned integer with arbitrary precision
- ``UINT8``: unsigned integer with 8-bit precision
- ``UINT16``: unsigned integer with 16-bit precision
- ``UINT32``: unsigned integer with 32-bit precision
- ``UINT64``: unsigned integer with 64-bit precision
- ``FLOATX``: floating point with arbitrary precision
- ``BFLOAT16``: brain floating point with 16-bit precisiion
- ``FLOAT16``: floating point with 16-bit precision
- ``FLOAT32``: floating point with 32-bit precision
- ``FLOAT64``: floating point with 64-bit precision
- ``BOOLEAN``: boolean data type
- ``INVALID``: invalid data type

Arbitrary Data Precision
************************

The arbitrary precision types are: ``INTX``, ``UINTX``, and ``FLOATX`` for signed integer, unsigned integer, and floating-point types respectively.
 
When a user writes eDSL code that defines a new constant scalar value, this value can take on one of the arbitrary precision types.
For instance:

.. code-block:: cpp

     Tensor zero = Tensor(0.0); // This will have a FLOATX type
     Tensor one = Tensor(1); // This will have a INTX type
 
Arbitrary precision types will be materialized to a concrete type when they come in contact with other concrete types. For instance:

.. code-block:: cpp

     Tensor A = Placeholder(DType::FLOAT16, {3, 3}); // create a 3x3xf16 input
     Tensor B = A + zero; // The data type of B will be 3x3xf16, because the ‘zero’ tensor gets materialized as a FLOAT16.
     Tensor C = A + one; // The data type of C will be 3x3xf16 as well.
 
Materialization is a function of type inference, it depends on other context clues to decide how to materialize a given type. There are certain situations where there isn’t enough context to make a decision on which type an arbitrary precision type should materialize into. In these cases an explicit cast is required. One such case is with the select op:

.. code-block:: cpp

     Tensor D = select(A < one, one, zero);
 
The problem here is that the select op can’t decide what the output type should be because both the true and false cases are arbitrary precision types. The eDSL will produce an error that advises the user to use an explicit cast.

.. code-block:: cpp

     Tensor zero = cast(Tensor{0}, A.dtype());
     Tensor one = cast(Tensor{1}, A.dtype());
     Tensor D = select(A < one, one, zero);
 
Another situation can occur with an assignment contraction:

.. code-block:: cpp

     TensorIndex i;
     Tensor one = Tensor{1};
     Tensor E = Contraction().outShape(3).outAccess(i).assign(one).build();
 
In this case, there’s not enough context to decide what the output type of the contraction. Again, the eDSL provides an error message because the output type is ambiguous. In this situation, the user will need to use an explicit cast.
 
