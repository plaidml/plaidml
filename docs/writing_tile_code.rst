=================
Writing Tile Code
=================
PlaidML uses a language called Tile to construct GPU kernels. When used to express machine learning operations, the Tile language is reasonably close to mathematical notation, while also being amenable to conversion into optimized GPU kernels.  In addition, all operations expressed in Tile can be be automatically differentiated. This tutorial is intended to help machine learning practitioners (or anyone with a background in software engineering and mathematics) get started using the Tile language.

Scope and Warning
-----------------

This tutorial provides an introduction to the Tile language. It is intended to help machine learning practitioners get started writing Tile code as quickly as possible, and as such covers core features, not every language detail. This is a tutorial, not a spec, and as such will consist of a series of examples, with a summary reference section at the end.

This tutorial covers how to write Tile code, not how Tile code is parsed and manipulated by PlaidML. It does not cover the workings of PlaidML utilities such as the Tile compiler or the Tile autodiffer.

Tile and PlaidML are still in early development and the Tile language is actively changing to add new functionality. While we do not expect the language features discussed here to radically change, we make no promises about the future specification of the Tile language. It is certain there will be substantive additions to the Tile language; it is likely that some minor changes will affect operations described in this tutorial; it is not impossible that there will be major changes to the Tile language affecting the contents of this tutorial.

How to Write Tile Code
----------------------

Sum Over Axis
=============

We're ready to look at some Tile code! Here's an operation that takes the sum over axis 0 of a 2D tensor (in Keras this would be ``K.sum(I, axis=0)``)::

    O[n: N] = +(I[m, n]);

An operation such as this which merges together values across one or more indices is called a *contraction*. The syntax may look a bit odd at first, but it's related to summation notation. Below we show how this Tile code is related to the mathematical formula for the operation by using colors to highlight corresponding pieces:

.. image:: docs/images/math-sum-0.png
    :height: 70pt
    :alt: O[n] = sum_m I[m, n]

.. image:: docs/images/code-sum-0.png
    :height: 40pt
    :alt: O[n: N] = +(I[m, n]);

In green, notice that the summation symbol is represented as ``+(...)`` in Tile code. In black, we have the portions of the notations that do not perfectly correspond. Here's why:
    - Summation notation includes a ``m`` subscript to indicate that ``m`` is the variable being summed over. Tile code implicitly sums over all valid indices (valid means not out of range for any tensor, and not failing any additional user-specified constraints as discussed in later examples).
    - Tile must be explicitly given the shape of any new tensor created, done in this code by ``: N``. In this case we want ``N`` to match the size of the last dimension of ``I`` (we'll see how that's done later when discussing how to write an entire Tile function). It is possible, however, to make this dimension of ``O`` larger or smaller, which would zero-pad or truncate ``O`` respectively. (For example, ``O[n: N+1] = +(I[m, n]);`` would result in a 0 as the last element of ``O`` if we're still assuming `N` is the size of the last dimension of ``I``).
    - Tile statements end in a semicolon.

Matrix Multiplication
=====================

Next we'll consider matrix multiplication. Let's look at the mathematical expression for the matrix multiplication ``C = AB`` written out in element-level detail:

.. image:: docs/images/math-mat-mul.png
    :height: 60pt
    :alt: C[i, j] = sum_k(A[i, k] * B[k, j])

We can convert this to Tile code using the same correspondence as the previous example: The summation sign becomes plus, the summation index is omitted, dimensions are given for the output tensor, and the statement ends in a semicolon. Here's the result::

    C[i, j: M, N] = +(A[i, k] * B[k, j]);

To have the dimensions correct, we need ``M`` to be the first dimension of ``A`` and ``N`` the last dimension of ``B``. Here's how this looks as part of a full Tile function::

    function (A[M, L], B[L, N]) -> (C) {
        C[i, j: M, N] = +(A[i, k] * B[k, j]);
    }

Notice that inputs have dimensions specified, and outputs do not. Inputs typically have dimensions declared; they can then be used in determining the dimension of intermediate and output tensors. Input dimensions can be repeated, which results in an error if the Tile function is passed inputs whose corresponding dimensions don't all have the specified size (for example ``A[N, N]`` would be constrained to a square).

Outputs do not have dimensions specified in the function header. Output dimensions are specified in the body of the function in the manner we saw above.

Max Over Axis
=============

Taking the maximum over axis 0 looks very similar to taking the sum over axis 0. Just like a sum is represented in Tile with ``+(...)``, a max is represented by ``>(...)``. Thus, the Tile code for max over axis 0 is just a single character change from sum over axis 0. Let's look at it as a Tile function::

    function (I[M, N]) -> (O) {
        O[n: N] = >(I[m, n]);
    }

Again, this corresponds closely to mathematical notation:

.. image:: docs/images/math-max-0.png
    :height: 60pt
    :alt: O[n] = max_m(I[m, n])

.. image:: docs/images/code-max-0.png
    :height: 40pt
    :alt: O[n: N] = >(I[m, n]);

Global Min
==========

There is a min contraction ``<(...)`` analogous to the max contraction ``>(...)``. For the purposes of this example, however, let's use the formula ``min(X) = -max(-X)``, to compute the min. We do this by combining a max computation with *elementwise* operations that perform the same operation (in this case negation) on every element of a tensor. Elementwise operations generally cannot be performed on the same line as contractions, so we write the global min function (for a 3D tensor) as follows::

    function (I) -> (O) {
        Neg = -I;
        O_Neg[] = >(Neg[i, j, k]);
        O = -O_Neg;
    }

There are several novel pieces in this example. First, note that the elementwise operations do not include dimensions. Dimensions are inferred from the inputs in elementwise operations, and so are never specified in elementwise ops. ``Neg`` has the same shape as ``I``, and ``O`` has the same shape as ``O_Neg``. When an elementwise binary operation is performed, the output shape is determined using `broadcasting semantics`_.

.. _broadcasting semantics: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

Which brings us to the next novelty: we have our first example of a 0D tensor, ``O_Neg``. Tensors in Tile are allowed to have zero dimensions. In such a case the tensor represents a scalar, i.e., a single value. In places where dimensions are specified, you can indicate a 0-dimensional tensor by using ``[]`` for the dimensions, as in this example.

Notice that we are taking the max over all axes in a single operation. Contractions implicitly aggregate over *all* indices that write to the same output location (in this case we aggregate over all values of ``i``, ``j``, and ``k``).

Notice also that every variable we assign to is new. This is not an accident or style choice, but a requirement of Tile. Tensors in Tile cannot be modified, only created and read, so every tensor used in a function must have a unique name. (The Tile compiler will figure out and handle memory reuse, in-place operations, etc by itself.)

Finally, note that ``I`` does not have dimensions listed in the function header, but ``Neg`` does have indices listed in the contraction. If none of the dimensions of an input tensor are needed in the body, they can be omitted in the header. (This is all-or-nothing; if you use one dimension of ``I``, you need to list all of them in the header.) Tensors used in a contraction, however, must always have all of their indices specified. Thus, any tensor used in a contraction always has a fixed number of dimensions (in this example 3). If you wanted a function that found the global max of a 2D tensor, you would need a different Tile function.

Average
=======

To compute the mean of a tensor, we need to sum the elements and divide by the total number of elements summed. We can do this by taking advantage of the fact that we can divide by a constant---including an input dimension---as an elementwise operation. Thus, to take the mean over axis 0 of a 2D tensor, we write ::

    function (I[X, Y]) -> (O) {
        Sum[y: Y] = +(I[x, y]);
        O = Sum / X;
    }

We can perform multiple elementwise operations on the same line, including operations on constants and input dimensions. So, while it would be possible to take a global mean of a 2D tensor in stages as so::

    function (I[X, Y]) -> (O) {
        Sum[] = +(I[x, y]);
        PartialMean = Sum / X;
        O = PartialMean / Y;
    }

it is more straightforward to merge the elementwise operations::

    function (I[X, Y]) -> (O) {
        Sum[] = +(I[x, y]);
        O = Sum / (X * Y);
    }


Max Pool 1D
===========

Next let's implement a size 2 stride 2 maxpool in Tile. This is the operation that splits a tensor into groups of 2 and takes the larger element from each group, yielding a tensor of half the original size. This is straightforward to implement in an imperative language by using for loops, e.g. in C::

    for (int i = 0; i < N/2; ++i) {
        float curr_max = FLT_MIN;
        for (int j = 0; j < 2; ++j) {
            if (I[2 * i + j] > curr_max) {
                curr_max = I[2 * i + j];
            }
        }
        O[i] = curr_max;
    }

(Note that this code, and the Tile code we'll be produce, both differ from the Keras ``K.pool*`` functions by ignoring channels and batches.)

For loops over tensor indices get translated into contractions when written in Tile. The most direct (and, sadly, wrong) implementation in Tile is ::

    function (I[N]) -> (O) {
        O[i: N / 2] = >(I[2 * i + j]);
    }

If you were to run this code, every entry of ``O`` would equal the global max of ``I``. We correctly determined that this was a maximization operation, and the indices for ``O`` and ``I`` match those used in the C code, so what went wrong?

The problem with this Tile code is that there are too many "valid" indices. For example, the case ``i = 1``, ``j = 3`` means that ``O[1]`` checks ``I[5]`` as one of the potential maximum values, even though ``O[1]`` is intended to be ``max(I[2], I[3])``. When we wrote the code with for loops, the inner loop restricted ``j`` to ``0`` or ``1``; in the Tile code, the compiler figured out the allowed values of ``j`` by looking at the shapes of the tensors, and the only restriction that imposes on ``j`` is that ``j`` must be an int satisfying ``0 <= 2 * i + j < N``.

The Tile language lets you add constraints to an operation to handle such cases. ::

    function (I[N]) -> (O) {
        O[i: N / 2] = >(I[2 * i + j]), j < 2;
    }

Something important to note here is that while we wrote ``j < 2``, this constraint actually means ``0 <= j < 2``. Constraints are always bounded below by 0, so to save typing ``0 <= `` is always omitted. (Without a constraint, however, index variables may still be negative: the original code included e.g. ``i = 1``, ``j = -1`` as valid index pair.)

We determined the Tile code for this example by starting from imperative code, but this Tile code is still very similar to mathematical notation, and we could have started there instead:

.. image:: docs/images/math-pool-1D.png
    :height: 60pt
    :alt: O[i] = max_(0 <= j < 2)(I[2i + j])

.. image:: docs/images/code-pool-1D.png
    :height: 30pt
    :alt: O[i: N / 2] = >(I[2 * i + j]), j < 2;

This Tile code handles odd values of ``N`` by rounding down the output tensor size. You may instead want to round up the output tensor size and use a smaller pool at the edge. This can be accomplished by simply adjusting the size of ``O``::

    function (I[N]) -> (O) {
        O[i: (N + 1) / 2] = >(I[2 * i + j]), j < 2;
    }

No special handling is needed for the case ``i = (N - 1) / 2``, ``j = 1``; this is out of range for ``I`` and so is ignored by Tile, which is exactly the intended behavior.

Valid Indices
=============
When discussing contractions, we've mentioned that they accumulate over "all valid indices". Hopefully the significance of this has been clear for the specific examples we've looked at, but to write complex or novel code it helps to have a precise understanding of what is meant by "valid indices". 

First, index validity is determined for a full set of index variables: ``j = 1`` is not valid or invalid as a standalone index value, but may be part of a valid or invalid set of index variables. For example, in the code ::

    O[i: (N + 1) / 2] = >(I[2 * i + j]), j < 2;

with ``N = 5``, the indices ``i = 1, j = 1`` are valid indices. However, ``i = 2, j = 1`` are not valid indices for this operation, nor are ``i = -1000, j = 1``.

A set of indices are *valid* if and only if:
    1. All the index variables are integers.
    2. All the index expressions for every tensor are in range. Specifically, if the index variable values are plugged into every index expression, all the resulting indices are non-negative integers less than the appropriate dimension.
    3. All the constraints are satisfied. Constraints always take the form ``[index expression] < [constant expression]`` (where ``[index expression]`` is a linear polynomial in the index variables and ``[constant expression]`` is a linear polynomial in the input dimensions), and they always implicitly include ``0 <= [index expression]``. Therefore we could also state this requirement as "every constraint's index expression is non-negative and less than its specified upper bound".

Skipping
========
The rule that all index variables must be integers allows us to "skip" certain otherwise valid entries. For example, consider the Tile function ::

    function (I[N, M]) -> (O) {
        O[2 * i: N] = +(I[2 * i, j]);
    }

This operation only writes to even entries of ``O``; while ``i = 1/2, j = 1`` does yield valid index expressions (``O[1]`` and ``I[1, 1]``), using a fractional index variable ``i`` makes these indices invalid. Note that some elements of ``O`` are never written to. Any unwritten elements in the output of a contraction are initialized to 0.

Cumulative Sum
==============
Suppose we want to take the cumulative sum of a 1D tensor. That is, we want ``O[i]`` to be the sum of all input entries ``I[k]`` where ``k <= i``. In summation notation, this is

.. image:: docs/images/math-cum-sum-raw.png
    :height: 70pt
    :alt: O[i] = sum_(k <= i) I[k]

However, we can't use ``k <= i`` as a constraint in Tile; all the index variables must be gathered into a single index expression on one side of the inequality. Thus, we rewrite this as ``0 <= i - k``. Since the 0 bound is implicitly included in all constraints, we just need to choose an upper bound large enough to never be hit. From the dimensions of the tensors, we already know ``i < N`` and ``0 <= k``, and so ``N`` is an appropriate upper bound. The resulting Tile code is ::

    function (I[N]) -> (O) {
        O[i: N] = +(I[k]), i - k < N;
    }

Alternatively, we could write ``k = i - j`` for ``j`` non-negative as an alternative way of forcing ``k`` to be no larger than ``i``. Then in summation notation we have

.. image:: docs/images/math-cum-sum-sub.png
    :height: 70pt
    :alt: O[i] = sum_(0 <= j) I[i - j]

and in Tile (noting that ``N`` is an upper bound for ``j`` that does not remove any valid indices)

.. image:: docs/images/code-cum-sum-sub.png
    :height: 30pt
    :alt: O[i: N] = +(I[i - j]), j < N;

Convolution
===========
We're now in a position to put together everything we've learned to write a complex Tile function: convolution. We'll first look at a relatively simple 1D convolution, then at a more complicated 2D convolution showing how to add functionality (dilation) that did not exist when PlaidML was first released (see `PlaidML issue #51`_). Even the 1D convolution is more complicated than anything we've done so far, but the Tile code we'll write will compile into optimized GPU kernels just as efficient as the ones we use in PlaidML's Keras backend.

.. _PlaidML issue #51: https://github.com/plaidml/plaidml/issues/51

(There's one exception to this, which is that PlaidML uses Winograd for certain kernels in the Keras backend. This uses an entirely different algorithmic approach which won't be covered in this tutorial. You can see how those convolutions work by looking at the ``_winograd`` function in the PlaidML Keras backend; if you do so, I recommend following along in `Lavin and Gray's paper`_ for using Winograd's algorithm for convolutions, as the algorithm is extremely opaque without an explanation of the math behind it.)

.. _Lavin and Gray's paper: https://arxiv.org/abs/1509.09308

Basic 1D Convolution
____________________
Let's implement a 1D convolution with output size equal to input size. This is implementing the Keras backend operation ::

    K.conv1d(x, kernel, padding='valid')

Let's start with the mathematical formula for this operation:

.. image:: docs/images/math-conv-1D-raw.png
    :height: 70pt
    :alt: O[n, x, c_o] = sum_k sum_(c_i) I[n, x + k, c_i] * K[k, c_i, c_o]

This is rather complicated, so let's walk through why this is the same convolution formula we're used to in machine learning.

A convolution produces output for a specific batch element at a specific location in a specific channel by taking a weighted sum of the input for that same batch element at that same location *and a surrounding region* over all input channels. The weights are given by ``K``, which depends on the output channel, the input channel, and the displacement within the input region relative to the reference location.

This generally matches the given formula: The output ``O`` is given as a sum of elements from the input ``I``, weighted by ``K``. Looking at the meaning of the index variables, we see that it matches exactly:
    - ``n`` represents which element of the batch we're on.
    - ``c_i`` represents which input channel we're on.
    - ``c_o`` represents which output channel we're on.
    - ``x`` represents our spatial location, giving the location being written to in ``O`` and the smallest element read from in ``I``.
    - Finally, ``k`` represents the kernel offset, that is, how far (in the spatial dimension) the input element we're reading is from the lower bound of the kernel.

This formula directly translates to Tile, although note that ``padding='valid'`` means that the spatial dimension of the output will be reduced by one less than the kernel size relative to the spatial dimension of the input:

.. image:: docs/images/math-conv-1D-color.png
    :height: 80pt
    :alt: O[n, x, c_o] = sum_k sum_(c_i) I[n, x + k, c_i] * K[k, c_i, c_o]

.. image:: docs/images/code-conv-1D-color.png
    :height: 60pt
    :alt: function (I[N, L, CI], K[LK, CI, CO]) -> (O) {O[n, x, co: N, L - LK + 1, CO] = +(I[n, x + k, ci] * K[k, ci, co]);}

Dilated 2D Convolution
______________________
We can tweak this general formula for a convolution to add various features, such as different strides, changing the padding, performing the convolution depthwise, etc. For this example, we will implement a dilated 2D convolution with dilation rate (2, 3). Specfically, we'll implement the Keras backend function ::

    K.conv2d(x, kernel, padding='valid', dilation_rate=(2, 3))

The formula for this is very similar to the previous convolution; we just have an additional spatial dimension for each tensor, and the kernel offset index variables are multiplied by dilation scaling factors when used to determine indices for ``I``:

.. image:: docs/images/math-dil-conv-2D.png
    :height: 60pt
    :alt: O[n, x, y, c_o] = sum_(k_x) sum_(k_y) sum_(c_i) I[n, x + 2k_x, y + 3k_y, c_i] * K[k_x, k_y, c_i, c_o]

The effective size for a dilated kernel with kernel size ``K`` and dilation rate ``d`` is ``d * (K - 1) + 1``, and so to achieve ``'valid'`` padding for this convolution, the x dimension must be reduced by ``2 * (LKx - 1)`` and the y dimension must be reduced by ``3 * (LKy - 1)``, where ``LKx`` and ``LKy`` are the x and y dimensions of the kernel respectively. The rest of the Tile code corresponds directly to the formula, and so we get ::

    function (I[N, Lx, Ly, CI], K[LKx, LKy, CI, CO]) -> (O) {
        O[n, x, y, co: N, Lx - 2 * (LKx - 1), Ly - 3 * (LKy - 1), CO] =
                +(I[n, x + 2 * kx, y + 3 * ky, ci] * K[kx, ky, ci, co]);
    }

Next Steps
----------
Now that you know how to write Tile code, you may want to learn how to wrap Tile code in Python to make it available to frontends like Keras or ONNX. The :doc:`adding_ops` tutorial will help you do this.

Reference
---------

Contractions
============
There are five *aggregation operations*:
    - Sum: ``+(...)``; when multiple values are computed for the same output location, they are added together.
    - Product: ``*(...)``; when multiple values are computed for the same output location, they are multiplied together.
    - Max: ``>(...)``; when multiple values are computed for the same output location, the largest one is used.
    - Min: ``<(...)``; when multiple values are computed for the same output location, the smallest one is used.
    - Assign: ``=(...)``; when multiple values are computed for the same output location, an error is raised. Note that the compiler errs on the side of caution and may raise an error even when no output location is assigned to multiple times. If the programmer manually confirms that there is at most one value computed for each output location, then any of the other aggregation operations will have equivalent behavior and can be used to bypass this error checking.

There are limited operations available inside a contraction. Principally, contractions allow the use of complex index expressions to determine which elements are read from a tensor. If there is only one tensor used in the contraction, such index manipulations are the only legal options. If there are two tensors used inside the contraction, you also choose a *combination operation* to determine how their values are combined. The only combination operations that are currently well-supported are multiplication (``*``) and addition (``+``).

Contractions aggregate over all sets of *valid indices*. A set of indices is valid for a contraction if and only if:
    - All index variables are integers
    - All index expressions used in tensors are within bounds
    - All user-specified constraints are satisfied

Elementwise Operations
======================
Elementwise operations never specify indices or dimensions. The shape of the output tensor is inferred from the shape of the input tensor(s). In most binary operations, if the input tensors have different shapes, the output shape is determined by broadcasting together the input shapes. If this is impossible or ambiguous, it is an error.

Common operations (not comprehensive; example tensor variable names provided to illustrate syntax):
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
    - Hyperbolic Tangent: ``O = tanh(A);``
    - Natural Log: ``O = log(A);``
    - Sigmoid: ``O = sigmoid(A);``
    - Conditional: ``O = C ? T : E;`` (``C`` may be a single value or a higher dimensional tensor to be evaluated elementwise. ``T`` and ``E`` must have the same shape, and unless ``C`` is known to be a constant at compile time, both will be evaluated.)

Variables
=========
Tile variables are one of three broad types:
    - *Tensors* (or *tensor variables*) are multidimensional arrays of a fixed shape. The scope of a tensor is the entire function. Tensors must begin with a capital letter.
    - *Dimensions* (or *dimension variables*) are positive integers initially passed to a function as sizes of input tensors. The scope of a dimension is the entire function. Dimensions must begin with a capital letter.
    - *Indices* (or *index variables*) are integers used in contractions to directly index a tensor or as part of a formula to compute a tensor index. The scope of an index is a single operation. Indices must begin with a lower case letter.

After the initial letter, all variable names may use letters, numbers, and ``_``.

Variable names may not be reused within the same scope; once a variable has been initialized, it cannot be modified.

Functions
=========
The function block syntax::

    function (I0[N0, N1], I1[M0, M1, M2], I2) -> (O0, O1) { ... }
