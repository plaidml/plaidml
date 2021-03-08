eDSL Examples
#############

Below are some examples that explore the various ways that the eDSL can be
customized to represent a wide variety of operations.

1D Max Pool
***********

Pooling is a method commonly used in neural networks to downsize a tensor. Like
the name implies, `pooling` is a type of contraction which groups elements of a
tensor together, then performs an aggregation on them. In this particular case,
we'll be looking at a 1D Max Pool, which is a native operation in most popular
frameworks:

.. tabs::

  .. group-tab:: TensorFlow

      .. code-block:: python

           tf.keras.layers.MaxPool1D(pool_size=2)

  .. group-tab:: PyTorch

      .. code-block:: python

           torch.nn.MaxPool1d(kernel_size=2)

Under the hood, this max pool splits a tensor into groups of 2 and takes the
larger element from each group, yielding a tensor of half the original size.
This is also quite straightforward to implement in C++/Python:

.. tabs:: 

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.cc
      :language: cpp
      :start-after: for_loop_max_pool_start
      :end-before: for_loop_max_pool_end
    
  .. group-tab:: Python

      .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.py
        :start-after: for_loop_max_pool_1d_start
        :end-before: for_loop_max_pool_1d_end


The ``for`` loop in the code above gets translated into a contraction when
written in the eDSL. Here is the eDSL code:

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.cc
      :language: cpp
      :start-after: max_pool_1d_start
      :end-before: max_pool_1d_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.py
      :pyobject: max_pool_1d

Notice that this code has a `constraint` added to it. 
Something important to note here is that while we wrote ``j < 2``, this
constraint actually means ``0<= j < 2``. Constraints are always bounded below
by ``0``.

.. math::

  \color{red}O[i]
  \color{default}=
  \color{green}\max_{\color{yellowgreen}0 \leq j < 2}
  \color{blue}I[2i + j]


.. math::
  
    \begin{aligned}
    \verb!Contraction()!
    &\verb!.outShape(N / 2)!\\
    &\color{red}\verb!.outAccess(i)!\\
    &\color{green}\verb!.max(!\\
    &\color{blue}\; \; \; \;\verb!  I(2 * i + j)!\\
    &\color{green}\; \; \; \; \; \; \; \; \verb!)!\\
    &\color{yellowgreen}\verb!.add_constraint(j < 2)!
    \end{aligned}


Why do we need this constraint in the first place? Let's prove that a
counterexample without a constraint would be incorrect:

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.cc
      :language: cpp
      :start-after: wrong_max_pool_start
      :end-before: wrong_max_pool_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/max_pool_1d.py
      :pyobject: wrong_max_pool_1d

If you were to run this code, every entry of ``O`` would equal the global max of
``I``. We correctly determined that this was a maximization operation, and the
indices for ``O`` and ``I`` match those used in the straight C++/Python code, 
so what went wrong?
The problem with this eDSL code is that there are too many "valid" indices. For
example, the case ``i = 1`` , ``j = 3`` means that ``O[1]`` checks ``I[5]`` as 
one of the potential maximum values, even though ``O[1]`` is intended to be 
``max(I[2], I[3])``.
When we wrote the code with for loops, the inner loop restricted ``j`` to ``0`` 
or ``1``; in the eDSL code, the compiler figured out the allowed values of 
``j`` by looking at the shapes of the tensors, and the only restriction that 
imposes on ``j`` is that ``j`` must be an integer satisfying ``0 <= 2 * i + j < 
N``.

1D Convolution
**************

Let's implement a 1D convolution with output size equal to input size (also
known as `valid` padding). Again, this operation is native to most of the
popular frameworks:

.. tabs::

  .. group-tab:: TensorFlow

      .. code-block:: python

           tf.keras.layers.Conv1D(filters, kernel_size, padding='valid')

  .. group-tab:: PyTorch

      .. code-block:: python

           torch.nn.Conv1D(in_channels, out_channels, kernel_size, padding=0)

Let's start with the mathematical formula for this operation:

.. math::

  \color{red}O[n, x, c_o]
  \color{default}=
  \color{green}\sum_k \sum_{c_i}
  \color{blue}I[n, x + k, c_i]
  \color{orange}\cdot
  \color{purple}K[k, c_i, c_o]

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

- ``n`` represents which element of the batch we're on.
- ``ci`` represents which input channel we're on.
- ``co`` represents which output channel we're on.
- ``x`` represents our spatial location, giving the location being written to in
  ``O`` and the smallest element read from in ``I``.
- Finally, ``k`` represents the kernel offset, that is, how far (in the spatial
  dimension) the input element we're reading is from the lower bound of the
  kernel.

This formula directly translates to eDSL, although note that ``padding='valid'``
means that the spatial dimension of the output will be reduced by one less than
the kernel size relative to the spatial dimension of the input:

.. math::

  \color{default}\verb!Contraction().outShape(O)!
  \color{red}\verb!.outAccess(n, x, co)!
  \color{green}\verb!.sum(!
  \color{blue}\verb!I(n, x + k, ci)!
  \color{orange}\verb! * !
  \color{purple}\verb!K(k, ci, co)!
  \color{green}\verb!)!


.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/conv_1d.cc
      :language: cpp
      :start-after: conv_1d_start
      :end-before: conv_1d_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/conv_1d.py
      :pyobject: conv_1d


Dilated 2D Convolution
**********************
We can tweak this general formula for a convolution to add various features,
such as different strides, changing the padding, performing the convolution
depthwise, etc. For this example, we will implement a dilated 2D convolution
with dilation rate (2, 3). Specfically, we'll implement the Keras backend
function:

.. tabs::
  
  .. group-tab:: TensorFlow

    .. code-block:: python

      O = tf.keras.layers.Conv2D(padding='valid', dilation_rate=(2, 3), input_shape)(I)
      
  .. group-tab:: PyTorch

    .. code-block:: python
    
      O = torch.nn.conv2d(in_channels, out_channels,kernel_size, dilation_rate=(2, 3))(I)



The formula for this is very similar to the previous convolution; we just have
an additional spatial dimension for each tensor, and the kernel offset index
variables are multiplied by dilation scaling factors when used to determine
indices for ``I``:

.. math::

  \color{red}O[n, x, y, c_o] \color{default}= \color{green}\sum_{k_x} \sum_{k_y} \sum_{c_i}
  \color{blue}I[\color{gray}n, x + 2k_x, y + 3k_y, c_i\color{blue}] \color{orange}*
  \color{purple}K[\color{gray}k_x, k_y, c_i, c_o\color{purple}]

The effective size for a dilated kernel with kernel size ``K`` and dilation rate
``d`` is ``d * (K - 1) + 1``, and so to achieve `'valid'` padding for this
convolution, the x dimension must be reduced by ``2 * (KX - 1)`` and the y
dimension must be reduced by ``3 * (KY - 1)``, where ``KX`` and ``KY`` are the 
x and y dimensions of the kernel respectively. The rest of the eDSL code
corresponds directly to the formula, and so we get:

.. math::
  \begin{aligned}
  \color{default}\verb!Contraction()! & \verb!.outShape(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO)!\\
  & \color{red}\verb!.outAccess(n, x, y, co)!\\
  & \color{green}\verb!.sum(!\\
  & \color{blue}\verb!    I(!\\
  & \color{gray}\verb!         n,!\\
  & \color{gray}\verb!         x + 2 * kx,!\\
  & \color{gray}\verb!         y + 3 * ky,!\\
  & \color{gray}\verb!         ci!\\
  & \color{blue}\verb!    )!\\
  & \color{orange}\verb!  * !
  \color{purple}\verb!K(!\\
  & \color{gray}\verb!         kx,!\\
  & \color{gray}\verb!         ky,!\\
  & \color{gray}\verb!         ci,!\\
  & \color{gray}\verb!         co!\\
  & \color{purple}\verb!    )!\\
  & \color{green}\verb!)!
  \end{aligned}

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/conv_2d_dilated.cc
      :language: cpp
      :start-after: conv_2d_dilated_start
      :end-before: conv_2d_dilated_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/conv_2d_dilated.py
      :pyobject: conv_2d_dilated

Complex Convolution
*******************
This example demonstrates a strided dilated padded grouped convolution.

.. math::

  \begin{aligned}
  \color{red}O&\color{red}[n, x_0, x_1, g, c_{o, g}] \cr
  &=\color{green}\sum_{k_0, k_1, c_{i, g}}
  (
    \color{blue}I[\color{gray}n, s_0 x_0 + d_0 k_0 - P_0, s_1 x_1 + d_1 k_1 - P_1, c_{i, g}\color{blue}] \color{orange}*
    \color{purple}K[\color{gray}k_0, k_1, g, c_{i, g}, c_{o, g}\color{purple}]
  )
  \end{aligned}

where ``s`` gives the stride coefficients, ``d`` gives the dilation
coefficients, and ``P`` gives the padding offsets.

.. math::
  \begin{aligned}
  \color{default}\verb!Contraction()! & \verb!.outShape(N, Y[0], Y[1], G, GCO)!\\
  & \color{red}\verb!.outAccess(n, x[0], x[1], g, gco)!\\
  & \color{green}\verb!.sum(!\\
  & \color{blue}\verb!    I(!\\
  & \color{gray}\verb!         n,!\\ 
  & \color{gray}\verb!         s[0] * x[0] + d[0] * k[0] - P[0],!\\
  & \color{gray}\verb!         s[1] * x[1] + d[1] * k[1] - P[1],!\\
  & \color{gray}\verb!         g,!\\
  & \color{gray}\verb!         gci!\\
  & \color{blue}\verb!    )!\\
  & \color{orange}\verb!  * !
  \color{purple}\verb!K(!\\
  & \color{gray}\verb!         k[0],!\\
  & \color{gray}\verb!         k[1],!\\
  & \color{gray}\verb!         g,!\\
  & \color{gray}\verb!         gci,!\\
  & \color{gray}\verb!         gco!\\
  & \color{purple}\verb!    )!\\
  & \color{green}\verb!)!
  \end{aligned}

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/complex_conv_2d.cc
      :language: cpp
      :start-after: complex_conv_start
      :end-before: complex_conv_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/complex_conv_2d.py
      :pyobject: complex_conv_2d


GEMV BLAS Level 2
*******************
This example illustrates matrix vector operations using the generalized matrix-vector (GEMV) multiplication of the form:

.. math::

   \color{red}O
   \color{default}  =
   \color{turquoise}\alpha
   \color{blue}A
   \color{purple}x
   \color{green}  +
   \color{turquoise}  \beta
   \color{purple}y

Here :math:`\color{blue}A` is a matrix, :math:`\color{purple}x` and :math:`\color{purple}y` 
are vectors and :math:`\color{turquoise}\alpha` and  :math:`\color{turquoise}\beta` are 
constants. 

Ignoring the constants at the moment, we can represent the matrix operation involved as:

.. math::

   \color{red}O[i, j]
   \color{default} = 
   \color{green}\sum_{i}
   \color{green} (
   \color{blue} A[
   \color{default}i, j
   \color{blue}]
   \color{orange} *
   {\color{purple} x[}
   \color{default}j
   {\color{purple}]}
   \color{green} )
   \color{magenta} +
   \color{purple} y[
   j
   \color{purple}]

This can easily be written in eDSL as follows. 

.. math::
    
    \verb!Contraction().outShape(I,J)!
    {\color{red}\verb!.outAccess(i,j)!}
    {\color{green}\verb!.sum(!}
    {\color{blue}\verb!A(!}
    \verb!i, j!
    {\color{blue}\verb!)!} 
    {\color{orange}\verb!*!} 
    {\color{purple}\verb!x(!}
    \verb!j!
    {\color{purple}\verb!)!}
    {\color{green}\verb!)!}
    {\color{magenta}\verb! + !}
    {\color{purple}\verb!y!}

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/gemv.cc
      :language: cpp
      :start-after: gemv_start
      :end-before: gemv_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/gemv.py
      :pyobject: gemv

Constant-vector multiplication and constant-tensor multiplication can be handled using the 
element-wise :math:`{\color{magenta}\verb!*!}` operator in eDSL. Thus :math:`\color{turquoise}\alpha` and  
:math:`\color{turquoise}\beta` can handled as follows. 

.. math::

  \begin{aligned}
  &\color{blue}\verb!A!
  {\color{magenta}\verb!  *  !}
  \color{turquoise}\verb!alpha!\\
  &\color{blue}\verb!y!
  {\color{magenta}\verb!  *  !}
  \color{turquoise}\verb!beta!
  \end{aligned}

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/gemv.cc
      :language: cpp
      :start-after: constant_gemv_start
      :end-before: constant_gemv_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/gemv.py
      :pyobject: gemv2

GEMM BLAS Level 3
*******************
This example illustrates matrix-matrix operations using the generalized matrix-matrix (GEMM) multiplication of the form:


.. math::

   \color{red}O
   \color{default}  =
   \color{turquoise} \alpha
   \color{blue}A
   \color{orange} *
   \color{blue} B
   \color{green} +
   \color{turquoise} \beta
   \color{blue}C


Here :math:`\color{blue}A` is a matrix, :math:`\color{purple}x` and :math:`\color{purple}y` 
are vectors and :math:`\color{turquoise}\alpha` and  :math:`\color{turquoise}\beta` are 
constants. 

We can represent the matrix operation involved as:

.. math::
   {\color{red}O[i, j]}
   \color{default}  =
   \color{green} \sum_{k} (
   \color{turquoise} \alpha
   \color{magenta}\cdot
   \color{blue}A[
   \color{default}i, k
   \color{blue}]
   \color{orange} *
   \color{blue} B[
   \color{default}k, j
   \color{blue}] 
   \color{green} )
   \color{magenta} +
   \color{turquoise} \beta
   \color{magenta}\cdot
   \color{blue}C[
   \color{default}i, j
   \color{blue}]


This can easily be written in eDSL as follows. 


.. math::

    \begin{aligned}
    &\color{blue}\verb! A!
    \color{default}\verb! =!
    \color{blue}\verb! A!
    \color{magenta}\verb! *!
    \color{turquoise}\verb! alpha! \\
    &\color{blue}\verb! C!
    \color{default}\verb! =!
    \color{blue}\verb! C!
    \color{magenta}\verb! *!
    \color{turquoise}\verb! beta! \\
    \verb!Contraction()!
    &\verb!.outShape(I,J)!\\
    &{\color{red}\verb!.outAccess(i,j)!}\\
    &{\color{green}\verb!.sum(!}\\
    &{\color{blue}\verb!     A(!}
    \verb!i,k!
    {\color{blue}\verb!)!}
    {\color{orange}\verb!  *  !}
    {\color{blue}\verb!B(!}
    \verb!k,j!
    {\color{blue}\verb!)!}\\
    &{\color{green}\verb! )!}\\
    &{\color{magenta}\verb!+!}
    {\color{blue}\verb!  C!}\\
    \end{aligned}

Where the contraction :math:`{\color{green}\verb!sum(...)!}`
handles the vector multiplication :math:`\color{blue}A \color{orange}* \color{blue}B`, the element wise operation 
:math:`{\color{magenta}\verb!+!}` handles the vector addition :math:`\color{green}(...)\color{magenta}+ \color{blue}C`.
The constant-vector multiplication is handled by an element wise :math:`{\color{magenta}\verb!*!}` operator. 

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/gemm.cc
      :language: cpp
      :start-after: gemm_start
      :end-before: gemm_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/gemm.py
      :pyobject: gemm


Uniform 8-bit Quantization
*******************

Quantization is a promising new approach used to reduce memory and compute resources required by neural network operations.
This example illustrates 8-bit uniform quantization operation :cite:`jacob2018quantization`:cite:`jain2020efficient`  in eDSL. 
The expected output is a quantized 8-bit tensor. This is accomplished using a simple technique illustrated below. 

.. math::

    \color{red} O
    \color{default} \: = \:
    \color{blue} A
    \color{magenta} \: / \:
    \color{turquoise} scale
    \color{magenta} \: + \:
    \color{turquoise} zeropoint

Which looks exactly the same in eDSL, :math:`\color{magenta}\verb! /!` and :math:`\color{magenta}\verb! +!` are element wise operations.

.. math::

    \color{pink}\verb! X!
    \color{default}\verb! =!
    \color{blue}\verb! A!
    \color{magenta}\verb! /!
    \color{turquoise}\verb! scale!

.. math::

    \color{red}\verb! O!
    \color{default}\verb! =!
    \color{pink}\verb! X!
    \color{magenta}\verb! +!
    \color{turquoise}\verb! zeropoint! 


:math:`\color{turquoise}\verb! scale!` and :math:`\color{turquoise}\verb! zeropoint!` are scaler quantities. Scale 
is real valued (float32 here) and zeropoint is the same type as the quantized tensor (int8 here).

.. tabs::

  .. group-tab:: C++

    .. literalinclude:: ../../plaidml/edsl/examples/quantize.cc
      :language: cpp
      :start-after: quantize_float32_int8_start
      :end-before: quantize_float32_int8_end

  .. group-tab:: Python

    .. literalinclude:: ../../plaidml/edsl/examples/quantize.py
      :pyobject: quantize_float32_to_int8


.. bibliography::
   :all:
