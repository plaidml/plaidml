Tile eDSL 
#############
The C++ Tile eDSL (Embedded Domain Specific Language) provides developers with a
way of describing a neural network so that the Stripe-based PlaidML compiler can
construct an efficient implementation.
This tutorial is intended to help machine learning practitioners (or anyone with
a background in software engineering and mathematics) get started using the C++
Tile eDSL.

Scope and Warning
*******************
This tutorial provides an introduction to the C++ Tile eDSL. It is intended to
help machine learning practitioners get started writing Tile code as quickly as
possible, and as such covers core features, not every language detail. This is a
tutorial, not a spec, and as such will consist of a series of examples, with a
summary reference section at the end.
This tutorial covers how to use the C++ Tile eDSL, not how Tile code is
constructed and manipulated by PlaidML. It does not cover the workings of
PlaidML utilities such as the pmlc compiler.
Tile and PlaidML are still being developed and the APIs discussed here are subject
to change.

How to Write Tile Code
************************

Sum Over Axis
================
We're ready to look at some C++ Tile code! Here's an operation that takes the
sum over axis `0` of a 2D tensor (in Keras this would be ``K.sum(I, axis=0)``):

.. tabs::

   .. group-tab:: C++

        .. code-block:: cpp

          Tensor sum_over_axis(const Tensor& I) {
            TensorDim M, N;
            TensorIndex m, n;
            I.bind_dims(M, N);
            auto O = TensorOutput(N);
            O(n) += I(m, n); // contraction
            return O;
          }

   .. group-tab:: Python

        .. code-block:: python

          def sum_over_axis(I):
            M, N = TensorDims(2)
            m, n = TensorIndexes(2)
            I.bind_dims(M, N)
            O = TensorOutput(N)
            # contraction
            O[n] += I[m, n]
            return O

An operation such as this which merges together values across one or more
indices is called a *contraction*. The syntax may look a bit odd at first, but
it's related to summation notation. Below we show how this C++ Tile code is
related to the mathematical formula for the operation by using colors to
highlight corresponding pieces:

.. math::

  \color{red}O[n]
  \color{default}=
  \color{green}\sum_{m}
  \color{blue}I[m, n]

.. math::

  \color{red}\verb|O(n)|
  \color{green}\verb| += |
  \color{blue}\verb|I(m, n)|\color{default}\verb|;|

In green, notice that the summation symbol is represented as ``+=`` in C++ Tile
code. Some portions of the notation do not perfectly correspond. Here's why:

- Summation notation includes a ``m`` subscript to indicate that ``m`` is the
  variable being summed over. Tile code implicitly sums over all valid indices
  (valid means not out of range for any tensor, and not failing any additional
  user-specified constraints as discussed in later examples).

- Tile must be explicitly given the shape of any new tensor created, done in
  this code by ``TensorOutput(N)``. In this case we want ``N`` to match the size of
  the last dimension of ``I``, which is specified by using ``I.bind_dims(M, N)``.
  It is possible, however, to make this dimension of ``O`` larger or smaller,
  which would zero-pad or truncate ``O`` respectively.
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

- As is the case for all C++ statements, they must end with a semicolon.

Max Over Axis
================
Taking the maximum over axis ``0`` looks very similar to taking the sum over axis
``0``. Just like a sum is represented in Tile with ``+=``, a max is represented by
``>=``. Thus, the Tile code for max over axis ``0`` is just a single character
change from sum over axis ``0``. Let's look at it as a Tile function:

.. tabs::

  .. group-tab:: C++

      .. code-block:: c++

        Tensor max_over_axis(const Tensor& I) {
          TensorDim M, N;
          TensorIndex m, n;
          I.bind_dims(M, N);
          auto O = TensorOutput(N);
          O(n) >= I(m, n);
          return O;
        }

  .. group-tab:: Python

      .. code-block:: python

            def max_over_axis(I):
              M, N = TensorDims(2)
              m, n = TensorIndexes(2)
              I.bind_dims(M, N)
              O = TensorOutput(N)
              O[n] >= I[m, n]
              return O

Again, this corresponds closely to mathematical notation:

.. math::

  \color{red}O[n]
  \color{default}=
  \color{green}\max_m
  \color{blue}I[m, n]

.. math::

  \color{red}\verb|O(n)|
  \color{green}\verb| >= |
  \color{blue}\verb|I(m, n)|\color{default}\verb|;|

Matrix Multiply
==================

Next we'll consider matrix multiplication. Let's look at the mathematical
expression for the matrix multiplication ``C = AB`` written out in element-level
detail:

.. math::

  C[i, j] = \sum_{k} (A[i, k] \cdot B[k, j])

We can convert this to C++ Tile code using the same correspondence as the
previous example: The summation sign becomes plus-assignment, the summation
index is omitted, dimensions are given for the output tensor, and the statement
ends in a semicolon. Here's the result:

.. tabs::

  .. group-tab:: C++

      .. code-block:: c++

        C(i, j) += A(i, k) * B(k, j);

  .. group-tab:: Python

      .. code-block:: python
      
        C[i, j] += A[i, k] * B[k, j];

To have correct dimensions, we need ``I`` to be the first dimension of ``A`` and ``J``
the last dimension of ``B``. Here's how this looks as part of a full Tile
function:

.. tabs::

  .. group-tab:: C++
  
    .. code-block:: c++

        Tensor matmul(const Tensor& A, const Tensor& B) {
          TensorDim I, J, K;
          TensorIndex i, j, k;
          A.bind_dims(I, K);
          B.bind_dims(K, J);
          auto C = TensorOutput(I, J);
          C(i, j) += A(i, k) * B(k, j);
          return C;
        }

  .. group-tab:: Python

    .. code-block:: python
    
        def matmul(A, B):
          I, J, K = TensorDims(3)
          i, j, k = TensorIndexes(3)
          A.bind_dims(I, K)
          B.bind_dims(K, J)
          C = TensorOutput(I, J)
          C[i, j] += A[i, k] * B[k, j]
          return C

Notice that we use ``bind_dims`` on inputs and we use ``TensorOutput`` on
outputs. Input dimensions can be repeated, which results in an error if the Tile
function is passed inputs whose corresponding dimensions don't all have the
specified size (for example `A.bind_dims(K, K)` would be constrained to a
square).

Global Min
=============
There is a min contraction ``<=`` analogous to the max contraction ``>=``. For the
purposes of this example, however, let's use the formula ``min(X) = -max(-X)``, to
compute the min. We do this by combining a max computation with *elementwise*
operations that perform the same operation (in this case negation) on every
element of a tensor. Elementwise operations generally cannot be performed on the
same line as contractions, so we write the global min function (for a 3D tensor)
as follows:

.. tabs::
  
  .. group-tab:: C++

      .. code-block:: c++

        Tensor global_min(const Tensor& I) {
          TensorIndex i, j, k;
          auto Neg = -I;
          auto O_Neg = TensorOutput();
          O_Neg() >= Neg(i, j, k);
          auto O = -O_Neg;
          return O;
        }

  .. group-tab:: Python

      .. code-block:: python
      
        def global_min(I):
          i, j, k = TensorIndexes(3)
          Neg = -I
          O_Neg = TensorOutput()
          O_Neg[()] >= Neg[i, j, k]
          O = -O_Neg
          return O


There are several novel pieces in this example. First, note that the elementwise
operations do not include dimensions. Dimensions are inferred from the inputs in
elementwise operations, and so are never specified in elementwise ops. `Neg` has
the same shape as ``I``, and ``O`` has the same shape as ``O_Neg``. When an
elementwise binary operation is performed, the output shape is determined using
`broadcasting semantics <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_.
Which brings us to the next novelty: we have our first example of a 0D tensor,
``O_Neg``. Tensors in Tile are allowed to have zero dimensions. In such a case the
tensor represents a scalar, i.e., a single value. In places where dimensions are
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

    .. code-block:: c++

      Tensor avg(const Tensor& I) {
        TensorDim X, Y;
        TensorIndex x, y;
        I.bind_dims(X, Y);
        auto Sum = TensorOutput();
        Sum(y) += I(x, y);
        return Sum / X;
      }

  .. group-tab:: Python

    .. code-block:: python

      def avg(I):
        X, Y = TensorDims(2)
        x, y = TensorIndexes(2)
        I.bind_dims(X, Y)
        Sum = TensorOutput()
        Sum[y] += I[x, y]
        return Sum / X

We can perform multiple elementwise operations on the same line, including
operations on constants and input dimensions. So, while it would be possible to
take a global mean of a 2D tensor in stages as so:

.. tabs::
  
  .. group-tab:: C++

    .. code-block:: c++

      Tensor avg(const Tensor& I) {
        TensorDim X, Y;
        TensorIndex x, y;
        I.bind_dims(X, Y);
        auto Sum = TensorOutput();
        Sum() += I(x, y);
        PartialMean = Sum / X;
        return PartialMean / Y;
      }

  .. group-tab:: Python

    .. code-block:: python

      def avg_stages(I):
        X, Y = TensorDims(2)
        x, y = TensorIndexes(2)
        I.bind_dims(X, Y)
        Sum = TensorOutput()
        Sum[()] += I[x, y]
        PartialMean = Sum / X
        return PartialMean / Y

it is more straightforward to merge the elementwise operations:

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

      Tensor avg(const Tensor& I) {
        TensorDim X, Y;
        TensorIndex x, y;
        I.bind_dims(X, Y);
        auto Sum = TensorOutput();
        Sum() += I(x, y);
        return Sum / (X * Y);
      }

  .. group-tab:: Python
    
    .. code-block:: python 
    
      def avg_merge(I):
        X, Y = TensorDims(2)
        x, y = TensorIndexes(2)
        I.bind_dims(X, Y)
        Sum = TensorOutput()
        Sum[()] += I[x, y]
        return Sum / (X * Y)

Max Pool 1D
==============

Next let's implement a size 2 stride 2 maxpool in Tile. This is the operation
that splits a tensor into groups of 2 and takes the larger element from each
group, yielding a tensor of half the original size. This is straightforward to
implement in straight C++:

.. tabs:: 

  .. group-tab:: C++

    .. code-block:: cpp

      float I[N], O[N / 2];
      for (int i = 0; i < N/2; ++i) {
        float curr_max = FLT_MIN;
        for (int j = 0; j < 2; ++j) {
          if (I[2 * i + j] > curr_max) {
            curr_max = I[2 * i + j];
          }
        }
        O[i] = curr_max;
      }
    
  .. group-tab:: Python

      .. code-block:: python

        for i in range (1 , N//2):
          curr_max = numpy.finfo(float).eps
          for j in range (1 , 2):
            if I[2*i*j] > curr_max:
              curr_max = I[2*i+j]
          O[i] = curr_max


``for`` loops over tensor indices get translated into contractions when written in
Tile. The most direct (and, sadly, wrong) implementation in Tile is:

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

        Tensor wrong_max_pool_1d(const Tensor& I) {
          TensorDim N;
          TensorIndex i, j;
          I.bind_dims(N);
          auto O = TensorOutput(N / 2);
          O(i) >= I(2 * i + j);
          return O;
        }

  .. group-tab:: Python

    .. code-block:: python

         def wrong_max_pool_1d(I):
            N = TensorDim()
            i, j = TensorIndexes(2)
            I.bind_dims(N)
            O = TensorOutput(N // 2)
            O[i] >= I[2 * i + j]
            return O

If you were to run this code, every entry of ``O`` would equal the global max of
``I``. We correctly determined that this was a maximization operation, and the
indices for ``O`` and ``I`` match those used in the straight C++ code, so what went wrong?
The problem with this Tile code is that there are too many "valid" indices. For
example, the case ``i = 1`` , ``j = 3`` means that ``O[1]`` checks ``I[5]`` as one of the
potential maximum values, even though ``O[1]`` is intended to be ``max(I[2], I[3])``.
When we wrote the code with for loops, the inner loop restricted ``j`` to ``0`` or
``1``; in the Tile code, the compiler figured out the allowed values of ``j`` by
looking at the shapes of the tensors, and the only restriction that imposes on
``j`` is that ``j`` must be an integer satisfying ``0 <= 2 * i + j < N``.
When can use ``add_constraint`` in Tile to handle such situations:

.. tabs::

.. global-tab:: C++

  .. code-block:: c++

    Tensor max_pool_1d(const Tensor& I) {
      TensorDim N;
      TensorIndex i, j;
      I.bind_dims(N);
      auto O = TensorOutput(N / 2);
      O(i) >= I(2 * i + j);
      O.add_constraint(j < 2);
      return O;
    }

  .. global-tab:: Python

    .. code-block:: python

      def max_pool_1d(I):
        N = TensorDim()
        i, j = TensorIndexes(2)
        I.bind_dims(N)
        O = TensorOutput(N // 2)
        O[i] >= I[2 * i + j]
        O.add_constraint(j < 2)
        return O

Something important to note here is that while we wrote ``j < 2``, this constraint
actually means ``0<= j < 2``. Constraints are always bounded below by ``0``.
(Without a constraint, however, index variables may still be negative: the
original code included e.g. ``i = 1``, ``j = -1`` as valid index pair.)
We determined the Tile code for this example by starting from imperative code,
but this Tile code is still very similar to mathematical notation, and we could
have started there instead:

.. math::

  \color{red}O[i]
  \color{default}=
  \color{green}\max_{\color{magenta}0 \leq j < 2}
  \color{blue}I[2i + j]

.. math::

  \begin{aligned}
  &
  \color{red}\verb|O(i)|
  \color{green}\verb| >= |
  \color{blue}\verb|I(2 * i + j)|\color{default}\verb|;|
  \cr
  &
  \color{default}\verb|O.add_constraint(|
  \color{magenta}\verb|j < 2|\color{default}\verb|);|
  \end{aligned}

This Tile code handles odd values of ``N`` by rounding down the output tensor
size. You may instead want to round up the output tensor size and use a smaller
pool at the edge. This can be accomplished by simply adjusting the size of ``O``:

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

      Tensor max_pool_1d(const Tensor& I) {
        TensorDim N;
        TensorIndex i, j;
        I.bind_dims(N);
        auto O = TensorOutput((N + 1) / 2);
        O(i) >= I(2 * i + j);
        O.add_constraint(j < 2);
        return O;
      }

  .. group-tab:: Python

    .. code-block:: python

      def max_pool_1d(I):
        N = TensorDim()
        i, j = TensorIndexes(2)
        I.bind_dims(N)
        O = TensorOutput((N + 1) // 2)
        O[i] >= I[2 * i + j]
        O.add_constraint(j < 2)
        return O

No special handling is needed for the case ``i = (N - 1) / 2``, ``j = 1``; this is
out of range for ``I`` and so is ignored by Tile, which is exactly the intended
behavior.

Valid Indices
=============
When discussing contractions, we've mentioned that they accumulate over "all
valid indices". Hopefully the significance of this has been clear for the
specific examples we've looked at, but to write complex or novel code it helps
to have a precise understanding of what is meant by "valid indices".
First, index validity is determined for a full set of index variables: ``j = 1``
is not valid or invalid as a standalone index value, but may be part of a valid
or invalid set of index variables. For example, in the code:

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

      I.bind_dims(N);
      auto O = TensorOutput((N + 1) / 2);
      O(i) >= I(2 * i + j);
      O.add_constraint(j < 2);
    
  .. group-tab:: Python

    .. code-block:: python

      I.bind_dims(N)
      O = TensorOutput[(N + 1) // 2];
      O[i] >= I[2 * i + j];
      O.add_constraint(j < 2);


with ``N = 5``, the indices ``i = 1``, ``j = 1`` are valid indices.
However, ``i = 2, j = 1`` are not valid indices for this operation, nor are ``i = -1000, j = 1``.
A set of indices are *valid* if and only if:

1. All the index variables are integers.

2. All the index expressions for every tensor are in range. Specifically, if the
   index variable values are plugged into every index expression, all the
   resulting indices are non-negative integers less than the appropriate
   dimension.

3. All the constraints are satisfied.
   Constraints always take the form ``[index expression] < [constant expression]``
   (where ``[index expression]`` is a linear polynomial in the index
   variables and ``[constant expression]`` is a linear polynomial in the input
   dimensions), and they always implicitly include ``0 <= [index expression]``.
   Therefore we could also state this requirement as "every constraint's index
   expression is non-negative and less than its specified upper bound".

Skipping
========
The rule that all index variables must be integers allows us to "skip" certain
otherwise valid entries. For example, consider the Tile function:

.. tabs::
  
  .. group-tab:: C++

    .. code-block:: c++

      Tensor skip(const Tensor& I) {
        TensorDim M, N;
        TensorIndex i, j;
        I.bind_dims(M, N);
        auto O = TensorOutput(N);
        O(2 * i) += I(2 * i, j);
        return O;
      }
  
  .. group-tab:: Python

    .. code-block:: python

        def skip(I):
          M, N = TensorDims(2)
          i, j = TensorIndexes(2)
          I.bind_dims(M, N)
          O = TensorOutput(N)
          O[2 * i] += I[2 * i, j]
          return O

This operation only writes to even entries of ``O``; while ``i = 1/2, j = 1`` does
yield valid index expressions (``O[1]`` and ``I[1, 1]``), using a fractional index
variable ``i`` makes these indices invalid. Note that some elements of ``O`` are
never written to. Any unwritten elements in the output of a contraction are
initialized to ``0``.

Cumulative Sum
==============
Suppose we want to take the cumulative sum of a 1D tensor. That is, we want
``O[i]`` to be the sum of all input entries ``I[k]`` where ``k <= i``. In summation
notation, this is:

.. math::

  O[i] = \sum_{k \leq i} I[k]

However, we can't use ``k <= i`` as a constraint in Tile; all the index variables
must be gathered into a single index expression on one side of the inequality.
Thus, we rewrite this as ``0 <= i - k``. Since the ``0`` bound is implicitly included
in all constraints, we just need to choose an upper bound large enough to never
be hit. From the dimensions of the tensors, we already know ``i < N`` and ``0 <= k``,
and so ``N`` is an appropriate upper bound. The resulting Tile code is:

.. tabs::

    .. group-tab:: C++

      .. code-block:: cpp

        Tensor csum(const Tensor& I) {
          TensorDim N;
          TensorIndex i, k;
          I.bind_dims(N);
          auto O = TensorOutput(N);
          O(i) += I(k);
          O.add_constraint(i - k < N);
          return O;
        }

    .. group-tab:: Python

      .. code-block:: python

        def csum(I):
          N = TensorDim()
          i, k = TensorIndexes(2)
          I.bind_dims(N)
          O = TensorOutput(N)
          O[i] += I[k]
          O.add_constraint(i - k < N)
          return O

Convolution
===========

Let's implement a 1D convolution with output size equal to input size. This is
implementing the Keras backend operation:

.. code-block:: python

  K.conv1d(x, kernel, padding='valid')

Let's start with the mathematical formula for this operation:

.. math::

  O[n, x, c_o] = \sum_k \sum_{c_i}(I[n, x + k, c_i] \cdot K[k, c_i, c_o])

This is rather complicated, so let's walk through why this is the same
convolution formula we're used to in machine learning.
A convolution produces output for a specific batch element at a specific
location in a specific channel by taking a weighted sum of the input for that
same batch element at that same location *and a surrounding region* over all
input channels. The weights are given by ``K``, which depends on the output
channel, the input channel, and the displacement within the input region
relative to the reference location.
This generally matches the given formula: The output ``O`` is given as a sum of
elements from the input ``I``, weighted by ``K``. Looking at the meaning of the
index variables, we see that it matches exactly:

- `n` represents which element of the batch we're on.
- `ci` represents which input channel we're on.
- `co` represents which output channel we're on.
- `x` represents our spatial location, giving the location being written to in
  `O` and the smallest element read from in `I`.
- Finally, `k` represents the kernel offset, that is, how far (in the spatial
  dimension) the input element we're reading is from the lower bound of the
  kernel.

This formula directly translates to Tile, although note that ``padding='valid'``
means that the spatial dimension of the output will be reduced by one less than
the kernel size relative to the spatial dimension of the input:

.. math::

  \color{red}O[n, x, c_o]
  \color{default}=
  \color{green}\sum_k \sum_{c_i}
  \color{blue}I[n, x + k, c_i]
  \color{orange}\cdot
  \color{lightblue}K[k, c_i, c_o]


.. math::

  \color{red}\verb|O(n, x, co)|
  \color{green}\verb| += |
  \color{blue}\verb|I(n, x + k, ci)|
  \color{orange}\verb| * |
  \color{lightblue}\verb|K(k, ci, co)|\color{default}\verb|;|


.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

      Tensor conv_1d(const Tensor& I, const Tensor& K) {
        TensorDim N, X, KX, CI, CO;
        TensorIndex n, x, k, ci, co;
        I.bind_dims(N, X, CI);
        K.bind_dims(KX, CI, CO);
        auto O = TensorOutput(N, X - KX + 1, CO);
        O(n, x, co) += I(n, x + k, ci) * K(k, ci, co);
        return O;
      }

  .. group-tab:: Python

    .. code-block:: python

        def conv_1d(I, K):
          N, X, KX, CI, CO = TensorDims(5)
          n, x, k, ci, co = TensorIndexes(5)
          I.bind_dims(N, X, CI)
          K.bind_dims(KX, CI, CO)
          O = TensorOutput(N, X - KX + 1, CO)
          O[n, x, co] += I[n, x + k, ci] * K[k, ci, co]
          return O


Dilated 2D Convolution
======================
We can tweak this general formula for a convolution to add various features,
such as different strides, changing the padding, performing the convolution
depthwise, etc. For this example, we will implement a dilated 2D convolution
with dilation rate (2, 3). Specfically, we'll implement the Keras backend
function:

.. code-block:: python

  K.conv2d(x, kernel, padding='valid', dilation_rate=(2, 3))


The formula for this is very similar to the previous convolution; we just have
an additional spatial dimension for each tensor, and the kernel offset index
variables are multiplied by dilation scaling factors when used to determine
indices for ``I``:

.. math::

  O[n, x, y, c_o] = \sum_{k_x} \sum_{k_y} \sum_{c_i}
  I[n, x + 2k_x, y + 3k_y, c_i] *
  K[k_x, k_y, c_i, c_o]

The effective size for a dilated kernel with kernel size ``K`` and dilation rate
``d`` is ``d * (K - 1) + 1``, and so to achieve `'valid'` padding for this
convolution, the x dimension must be reduced by ``2 * (KX - 1)`` and the y
dimension must be reduced by ``3 * (KY - 1)``, where ``KX`` and ``KY`` are the x and y
dimensions of the kernel respectively. The rest of the Tile code corresponds
directly to the formula, and so we get:

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++

      Tensor conv_2d(const Tensor& I, const Tensor& K) {
        TensorDim N, X, Y, KX, KY, CI, CO;
        TensorIndex n, x, y, kx, ky, ci, co;
        I.bind_dims(N, X, Y, CI);
        K.bind_dims(KX, KY, CI, CO);
        auto O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO);
        O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
        return O;
      }

  .. group-tab:: Python

    .. code-block:: python
    
        def conv_2d_dilated(I, K):
          N, X, Y, KX, KY, CI, CO = TensorDims(7)
          n, x, y, kx, ky, ci, co = TensorIndexes(7)
          I.bind_dims(N, X, Y, CI)
          K.bind_dims(KX, KY, CI, CO)
          O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO)
          O[n, x, y, co] += I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]
          return O

Complex Convolution
===================
This final example demonstrates a strided dilated padded grouped convolution.

.. math::

  \begin{aligned}
  O&[n, x_0, x_1, g, c_{o, g}] \cr
  &=\sum_{k_0, k_1, c_{i, g}}
  (
    I[n, s_0 x_0 + d_0 k_0 - P_0, s_1 x_1 + d_1 k_1 - P_1, c_{i, g}] *
    K[k_0, k_1, g, c_{i, g}, c_{o, g}]
  )
  \end{aligned}

where *`s`* gives the stride coefficients, *`d`* gives the dilation
coefficients, and *`P`* gives the padding offsets.

.. tabs::

  .. group-tab:: C++

    .. code-block:: c++
        
        Tensor complex_conv_2d(
          const Tensor& I,
          const Tensor& K,
          const std::vector<size\_t>& s,  // stride coeffs
          const std::vector<size\_t>& d   // dilation coeffs
        ) {
            // "same-lower" autopadding will be applied
            TensorDim N, G, GCI, GCO;
            std::vector<TensorDim> X(2);
            std::vector<TensorDim> K(2);
            TensorIndex n, g, gci, gco;
            std::vector<TensorIndex> x(2);
            std::vector<TensorIndex> k(2);
            I.bind_dims(N, X[0], X[1], G, GCI);
            K.bind_dims(K[0], K[1], G, GCI, GCO);
            // Compute output spatial dimensions
            std::vector<TensorDim> Y(2);
            for (size_t i = 0; i < Y.size(); ++i) {
              Y[i] = (X[i] + s[i] \- 1) / s[i];
            }
            // Compute the effective kernel size after dilation
            std::vector<TensorDim> EK(2);
            for (size_t i = 0; i < EK.size(); ++i) {
              EK[i] = d[i] \* (K[i] \- 1) + 1;
            }
            // Compute the padding offset
            std::vector<TensorDim> P(2);
            for (size_t i = 0; i < P.size(); ++i) {
              P[i] = ((Y[i] \- 1) \* s[i] + EK[i] \- X[i]) / 2;
            }
            // Specify the output size
            auto O = TensorOutput(N, Y0, Y1, G, GCO);
            // Compute the convolution
            O(n, x[0], x[1], g, gco) +=
              I(n, s[0]\*x[0] + d[0]\*k[0] \- P[0], s[1]\*x[1] + d[1]\*k[1] \- P[1], g, gci) \*
              K(k0, k1, g, gci, gco);
            return O;
        }

  .. group-tab:: Python

    .. code-block:: python

        def complex_conv_2d(
            I,
            K,
            s0,
            s1,  # stride coeffs
            d0,
            d1  # dilation coeffs
            ):
                # "same-lower" autopadding will be applied
                N, G, GCI, GCO = TensorDims(4)
                X0, X1 = TensorDims(2)
                K0, K1 = TensorDims(2)
                n, g, gci, gco = TensorIndexes(4)
                x0, x1 = TensorIndexes(2)
                k0, k1 = TensorIndexes(2)
                I.bind_dims(N, X0, X1, G, GCI)
                K.bind_dims(K0, K1, G, GCI, GCO)

                # Compute output spatial dimensions
                Y0, Y1 = TensorDims(2)
                Y0 = (X0 + s0 - 1) // s0
                Y1 = (X1 + s1 - 1) // s1

                #Compute the effective kernel size after dilation
                EK0, EK1 = TensorDims(2)
                EK0 = d0 * (K0 - 1) + 1
                EK1 = d1 * (K1 - 1) + 1

                #Compute the padding offset
                P0, P1 = TensorDims(2)
                P0 = ((Y0 - 1) * s0 + EK0 - X0) // 2
                P1 = ((Y1 - 1) * s1 + EK1 - X1) // 2

                # Specify the output size
                O = TensorOutput(N, Y0, Y1, G, GCO)

                # Compute the convolution
                O[n, x0, x1, g, gco] += I[n, s0 * x1 + d0 * k0 - P0, s1 * x1 + d1 * k1 -
                                          P1, g, gci] * K[k0, k1, g, gci, gco]
                return O



Reference
*********

Contractions
============

There are five *aggregation* operations:

- `operator +=` or `sum`: When multiple values are computed for the same
  output location, they are added together.
- `operator *=` or `product`: when multiple values are computed for the same
  output location, they are multiplied together.
- `operator >=` or `max`: when multiple values are computed for the same
  output location, the largest one is used.
- `operator <=` or `min`: when multiple values are computed for the same
  output location, the smallest one is used.
- `operator =` or `assign`: when multiple values are computed for the same
  output location, an error is raised. Note that the compiler errs on the side
  of caution and may raise an error even when no output location is assigned to
  multiple times. If the programmer manually confirms that there is at most one
  value computed for each output location, then any of the other aggregation
  operations will have equivalent behavior and can be used to bypass this error
  checking.

There are limited operations available inside a contraction. Principally,
contractions allow the use of complex index expressions to determine which
elements are read from a tensor. If there is only one tensor used in the
contraction, such index manipulations are the only legal options. If there are
two tensors used inside the contraction, you also choose a *combination*
operation to determine how their values are combined. The only combination
operations that are currently well-supported are multiplication (`*`) and
addition (`+`).
Contractions aggregate over all sets of *valid indices*. A set of indices is
valid for a contraction if and only if:

- All index variables are integers
- All index expressions used in tensors are within bounds
- All user-specified constraints are satisfied

Elementwise Operations
======================
Elementwise operations never specify indices or dimensions. The shape of the
output tensor is inferred from the shape of the input tensor(s). In most binary
operations, if the input tensors have different shapes, the output shape is
determined by broadcasting together the input shapes. If this is impossible or
ambiguous, it is an error.
Common operations (not comprehensive; example tensor variable names provided to
illustrate syntax):

- Addition: `O = A + B;`
- Subtraction: `O = A - B;`
- Multiplication: `O = A * B;`
- Division: `O = A / B;`
- Equality: `O = A == B;`
- Inequality: `O = A != B;`
- Less: `O = A < B;`
- Square Root: `O = sqrt(A);`
- Exponential: `O = exp(A);`
- Power: `O = pow(A, B);`
- Sine: `O = sin(A);`
- Hyperbolic Tangent: `O = tanh(A);`
- Natural Log: `O = log(A);`
- Sigmoid: `O = sigmoid(A);`
- Conditional: `O = select(C, T, F);` (`C` may be a single value or a higher
  dimensional tensor to be evaluated elementwise. `T` and `F` must have the same
  shape, and unless `C` is known to be a constant at compile time, both will be
  evaluated.)

Types
=====

- `Tensor`: Multidimensional arrays of a fixed shape. The scope of a tensor is
  the entire function. By convention, tensors begin with a capital letter.
- `TensorDim`: Positive integers initially passed to a function as sizes of
  input tensors. The scope of a dimension is the entire function. By convention,
  dimensions begin with a capital letter.
- `TensorIndex`: Symbolic integers used in contractions to directly index a
  tensor or as part of a formula to compute a tensor index. The scope of an
  index is a single operation. By convention, indices begin with a lower case
  letter.