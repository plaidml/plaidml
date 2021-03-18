eDSL Operations 
###############
The eDSL has three main types of operations: contractions, elementwise, 
and specials.

- :ref:`Contraction operations<contractions>` are best described as 
  aggregations that are performed over a sequence of numbers.
- :ref:`Elementwise operations<eltwise>` are performed on each individual 
  element(s) of the input tensor(s), resulting in an output that has the same 
  shape as the input.
- :ref:`Special operations<specials>` are those which have access patterns or 
  behaviors that are somehow unique from contractions and elementwise operations.

This tutorial will walk through each of these types of operations and 
demonstrate their usage.

.. _contractions:

Contraction Operations
************************
Contractions are built up from a few key components: `index expressions`,
`constraints`, `combinations`, and `aggregations`.

Contractions allow the use of complex `index expressions` to determine which
elements are read from a tensor. If there is only one tensor used in the
contraction, such index manipulations are the only legal options. A set of
indices is valid for a contraction if and only if: 

- All index variables are integers
- All index expressions used in tensors are within bounds
- All user-specified `constraints` are satisfied

`Constraints` always take the form ``[index expression] < [constant expression]``
(where ``[index expression]`` is a linear polynomial in the index variables and
``[constant expression]`` is a linear polynomial in the input dimensions), and
they always implicitly include ``0 <= [index expression]``. We'll show a few
examples of constraints later in this document.

Therefore, we could also state the `index expression` requirement as: "Every
constraint’s index expression is non-negative and less than its specified upper
bound."

If there are multiple tensors used inside a contraction, you have the ability to
choose a `combination` operation to determine how their values are combined. The
only combination operations that are currently well-supported are multiplication
(``*``) and addition (``+``). 

Contractions `aggregate` over all sets of valid indices. There are five
different types of aggregations that a Contraction can perform. 

- ``sum``: When multiple values are computed for the same output location, they
  are added together.
- ``product``: When multiple values are computed for the same output location, 
  they are multiplied together.

- ``max``: When multiple values are computed for the same output location, the 
  largest one is used.

- ``min``: When multiple values are computed for the same output location, the 
  smallest one is used.

- ``assign``: When multiple values are computed for the same output location, 
  an error is raised. Note that the compiler errs on the side of caution and may 
  raise an error even when no output location is assigned to multiple times. If 
  the programmer manually confirms that there is at most one value computed for 
  each output location, then any of the other aggregation operations will have 
  equivalent behavior and can be used to bypass this error checking.

Now that we've gone over the key components that build up a contraction, let's
walk through a few simple examples. Additional examples are available in the `eDSL Examples page`_.

Sum Over Axis
================
The first contraction we'll look at is a summation aggregation, performed on 
axis `0` of a 2D tensor (in Numpy this would be ``np.sum(I, axis=0)``):

.. tabs::

   .. group-tab:: C++

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
        :language: cpp
        :start-after: sum_over_axis_start
        :end-before: sum_over_axis_end

   .. group-tab:: Python

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
        :pyobject: sum_over_axis

As a rule of thumb, a ``Contraction`` merges together values across one or more
indices. Below we compare the eDSL code to the mathematical formula for 
the operation by using colors to highlight corresponding pieces:

.. math::

  \color{red}O[n]
  \color{default}=
  \color{green}\sum_{m}
  \color{blue}I[m, n]

.. math::
  \color{default}\verb!Contraction().outShape(N)!
  \color{red}\verb!.outAccess(n)!
  \color{green}\verb!.sum(!
  \color{blue}\verb!I[m, n]!
  \color{green}\verb!)!

In green, notice that the summation symbol is represented using ``.sum()`` in 
eDSL code. In blue, notice that the input ``I`` is indexed using ``m`` and 
``n``. Some portions of the notation do not perfectly correspond. Here's why:

- Summation notation includes a ``m`` subscript to indicate that ``m`` is the
  variable being summed over. eDSL code implicitly sums over all valid indices
  (valid means not out of range for any tensor, and not failing any additional
  user-specified constraints as discussed in later examples).

- eDSL must be explicitly given the shape of any new tensor created, done in
  this code by ``Contraction().outShape(N)``. In this case we want ``N`` to
  match the size of the last dimension of ``I``, which is specified by using
  ``I.bind_dims(M, N)``. It is possible, however, to make this dimension of
  ``O`` larger or smaller, which would zero-pad or truncate ``O`` respectively.
  For example,

  .. tabs::

    .. group-tab:: C++

        .. code-block:: c++

            auto O = TensorOutput(N + 1);

    .. group-tab:: Python

        .. code-block:: python
        
            O = TensorOutput(N+1)
      
  would result in a `0` as the last element of `O` if we're still assuming `N`
  is the size of the last dimension of `I`.


Max Over Axis
================
Taking the maximum over axis ``0`` looks very similar to taking the sum over
axis ``0``. Just like a sum is represented in eDSL with ``.sum()``, a max is
represented by ``.max()``. Thus, to perform a max over axis ``0``, we can take
the same exact code for sum over axis ``0``, and switch ``.sum()`` to
``.max()``. Let's look at it as a eDSL function:

.. tabs::

  .. group-tab:: C++

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
        :language: cpp
        :start-after: max_over_axis_start
        :end-before: max_over_axis_end

  .. group-tab:: Python

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
        :pyobject: max_over_axis

Again, this corresponds closely to mathematical notation:

.. math::

  \color{red}O[n]
  \color{default}=
  \color{green}\max_m
  \color{blue}I[m, n]

.. math::

  \color{default}\verb!Contraction().outShape(N)!
  \color{red}\verb!.outAccess(n)!
  \color{green}\verb!.max(!
  \color{blue}\verb!I[m, n]!
  \color{green}\verb!)!

Matrix Multiply
==================

Next we'll consider matrix multiplication. Let's look at the mathematical
expression for the matrix multiplication ``C = AB`` written out in element-level
detail:

.. math::

  C[i, j] = \sum_{k} (A[i, k] \cdot B[k, j])

We can convert this to eDSL code using the same correspondence as the
previous example: The summation sign becomes plus-assignment, the summation
index is omitted, dimensions are given for the output tensor, and the statement
ends in a semicolon. Here's the result:

.. math::

  \color{default}\verb!Contraction().outShape(C)!
  \color{red}\verb!.outAccess(i, j)!
  \color{green}\verb!.sum(!
  \color{blue}\verb!A[i, k] * B[k, j]!
  \color{green}\verb!)!

To have correct dimensions, we need ``I`` to be the first dimension of ``A`` 
and ``J`` the last dimension of ``B``. Here's how this looks as part of a full 
eDSL
function:

.. tabs::

  .. group-tab:: C++
  
    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
        :language: cpp
        :start-after: matmul_start
        :end-before: matmul_end

  .. group-tab:: Python
    
      .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
        :pyobject: matmul

Notice that we use ``bind_dims`` on inputs and we use ``TensorOutput`` on
outputs. Input dimensions can be repeated, which results in an error if the eDSL
function is passed inputs whose corresponding dimensions don't all have the
specified size (for example `A.bind_dims(K, K)` would be constrained to a
square).

Global Min
=============
There is a min contraction ``<=`` analogous to the max contraction ``>=``. For 
the
purposes of this example, however, let's use the formula ``min(X) = -max(-X)``, 
to
compute the min. We do this by combining a max computation with *elementwise*
operations that perform the same operation (in this case negation) on every
element of a tensor. Elementwise operations generally cannot be performed on the
same line as contractions, so we write the global min function (for a 3D tensor)
as follows:

.. tabs::
  
  .. group-tab:: C++

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
        :language: cpp
        :start-after: global_min_start
        :end-before: global_min_end

  .. group-tab:: Python

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
        :pyobject: global_min


There are several novel pieces in this example. First, note that the elementwise
operations do not include dimensions. Dimensions are inferred from the inputs in
elementwise operations, and so are never specified in elementwise ops. `Neg` has
the same shape as ``I``, and ``O`` has the same shape as ``O_Neg``. When an
elementwise binary operation is performed, the output shape is determined using
:ref:`broadcasting semantics <broadcasting-semantics>`_.
Which brings us to the next novelty: we have our first example of a 0D tensor,
``O_Neg``. Tensors in eDSL are allowed to have zero dimensions. In such a case 
the tensor represents a scalar, i.e., a single value. In places where 
dimensions are
specified, you can indicate a 0-dimensional tensor by using ``()`` for the
dimensions, as in this example.
Notice that we are taking the max over all axes in a single operation.
Contractions implicitly aggregate over *all* indices that write to the same
output location (in this case we aggregate over all values of ``i``, ``j``, and
``k``).

Average
==========
To compute the mean of a tensor, we need to sum the elements and divide by the
total number of elements summed. We can do this by taking advantage of the fact
that we can divide by a constant (including an input ``TensorDim``) as an
elementwise operation. Thus, to take the mean over axis ``0`` of a 2D tensor, we
write:

.. tabs::
  
  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: avg_start
      :end-before: avg_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: avg

We can perform multiple elementwise operations on the same line, including
operations on constants and input dimensions. So, while it would be possible to
take a global mean of a 2D tensor in stages as so:

.. tabs::
  
  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: avg_stages_start
      :end-before: avg_stages_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: avg_stages

it is more straightforward to merge the elementwise operations:

.. tabs::

  .. group-tab:: C++

   .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: avg_merge_start
      :end-before: avg_merge_end

  .. group-tab:: Python
    
    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: avg_merge

Skipping
========
The rule that all index variables must be integers allows us to "skip" certain
otherwise valid entries. For example, consider the eDSL function:

.. tabs::
  
  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: skip_start
      :end-before: skip_end
  
  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: skip

This operation only writes to even entries of ``O``; while ``i = 1/2`` , ``j = 
1`` does yield valid index expressions (``O[1]`` and ``I[1, 1]``), using a 
fractional 
index variable ``i`` makes these indices invalid. Note that some elements of 
``O`` are
never written to. Any unwritten elements in the output of a contraction are
initialized to ``0``.

Cumulative Sum
==============
Suppose we want to take the cumulative sum of a 1D tensor. That is, we want
``O[i]`` to be the sum of all input entries ``I[k]`` where ``k <= i``. In 
summation notation, this is:

.. math::

  O[i] = \sum_{k \leq i} I[k]

However, we can't use ``k <= i`` as a constraint in eDSL; all the index 
variables must be gathered into a single index expression on one side of the 
inequality.
Thus, we rewrite this as ``0 <= i - k``. Since the ``0`` bound is implicitly 
included in all constraints, we just need to choose an upper bound large enough 
to never
be hit. From the dimensions of the tensors, we already know ``i < N`` and ``0 
<= k``, and so ``N`` is an appropriate upper bound. The resulting eDSL code is:

.. tabs::

    .. group-tab:: C++

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
        :language: cpp
        :start-after: cumsum_start
        :end-before: cumsum_end

    .. group-tab:: Python

      .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
        :pyobject: cumsum

.. _eltwise:

Elementwise Operations
**********************
Elementwise operations never specify indices or dimensions. The shape of the
output tensor is inferred from the shape of the input tensor(s). In most binary
operations, if the input tensors have different shapes, the output shape is
determined by :ref:`broadcasting together the input 
shapes<broadcast-semantics>`. If this is impossible or ambiguous, it is an 
error.
Common operations (not comprehensive; example tensor variable names provided to
illustrate syntax):

- Addition: ``O = A + B;``
- Subtraction: ``O = A - B;``
- Multiplication: ``O = A * B;``
- Division: ``O = A / B;``
- Equality: ``O = A == B;``
- Inequality: ``O = A != B;``
- Less: ``O = A < B;``
- Square Root: ``O = sqrt(A);``
- Exponential: ``O = exp(A);``
- Power: ``O = pow(A, B);``
- Sine: ``O = sin(A);``
- Cosine: ``O = cos(A);``
- Hyperbolic Tangent: ``O = tanh(A);``
- Natural Log: ``O = log(A);``
- Sigmoid: ``O = sigmoid(A);``
- Conditional: ``O = select(C, T, F);`` (Note: ``C`` may be a single value or a
  higher dimensional tensor to be evaluated elementwise. ``T`` and ``F`` must
  have the same shape, and unless ``C`` is known to be a constant at compile
  time, both will be evaluated.)

.. _specials:

Special Operations
******************
- Gather
- Index
- Portable Random Number Generation (PRNG)
- Scatter
- Layer

In some cases, it may be useful to specify an arbitrary block of operations as a
single layer function. The ``layer`` operation provides a convenient method for 
binding ``Tensor`` inputs and (if applicable) function attributes to a function 
definition:

.. tabs::
  
  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: layer_start
      :end-before: layer_end
  
  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: layer

- Trace

While debugging eDSL code it can often be desirable to insert print statements to
understand what the program is doing as it executes. However, typical print
statements are not generally useful for this, as execution of eDSL occurs deep within
the PlaidML backend. The ``trace`` operation allows print statements to be specified
within the eDSL program and printed during the program's execution. Additionally, if
multiple tracepoints are specified, the time elapsed between consecutive tracepoints
will printed automatically.

.. tabs::
  
  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.cc
      :language: cpp
      :start-after: trace_start
      :end-before: trace_end
  
  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/edsl_docs.py
      :pyobject: trace

.. _broadcast-semantics:

Operation Broadcasting Semantics
********************************

Automatic Broadcasting
======================
The eDSL automatically attempts to broadcast the input ``Tensors`` of each 
operation in order to obtain valid input shapes. To do this, it follows `Numpy 
broadcasting semantics`_.

Manual Broadcasting
===================
In some use cases, manual broadcasting may be required.

For example, ...

This type of broadcast is perfectly valid, but it does not follow Numpy 
broadcasting semantics.

.. _eDSL Examples page: edsl_examples.html
.. _Numpy broadcasting semantics: https://numpy.org/doc/stable/user/basics.broadcasting.html
