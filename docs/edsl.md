# C++ Tile EDSL

The C++ Tile EDSL (Embedded Domain Specific Language) provides developers with a
way of describing a neural network so that the Stripe-based PlaidML compiler can
construct an efficient implementation.

This tutorial is intended to help machine learning practitioners (or anyone with
a background in software engineering and mathematics) get started using the C++
Tile EDSL.

## Scope and Warning

This tutorial provides an introduction to the C++ Tile EDSL. It is intended to
help machine learning practitioners get started writing Tile code as quickly as
possible, and as such covers core features, not every language detail. This is a
tutorial, not a spec, and as such will consist of a series of examples, with a
summary reference section at the end.

This tutorial covers how to use the C++ Tile EDSL, not how Tile code is
constructed and manipulated by PlaidML. It does not cover the workings of
PlaidML utilities such as the pmlc compiler.

Tile and PlaidML are still being developed and the APIs discussed here are subject
to change.

## How to Write Tile Code

### Sum Over Axis

We're ready to look at some C++ Tile code! Here's an operation that takes the
sum over axis `0` of a 2D tensor (in Keras this would be `K.sum(I, axis=0)`):

```c++
Tensor sum_over_axis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) += I(m, n); // contraction
  return O;
}
```

An operation such as this which merges together values across one or more
indices is called a _contraction_. The syntax may look a bit odd at first, but
it's related to summation notation. Below we show how this C++ Tile code is
related to the mathematical formula for the operation by using colors to
highlight corresponding pieces:

<!-- ```math
 \Large
 \textcolor{red}{O[n]}
 \textcolor{yellow}{=}
 \textcolor{green}{\sum_{m}}
 \textcolor{cyan}{I[m, n]}
 ``` -->

 ![\Large \color{red}O\[n\] \color{black}=\color{green}\sum_{m}{\color{blue}I\[m,n\]}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7DO%5Bn%5D%20%5Ccolor%7Bblack%7D%3D%5Ccolor%7Bgreen%7D%5Csum_%7Bm%7D%7B%5Ccolor%7Bblue%7DI%5Bm%2Cn%5D%7D)

<!-- ```math
 \Large
 \texttt{
   \textcolor{red}{O(n)}
   \textcolor{green}{+=}
   \textcolor{cyan}{I(m, n)};
 }
 ``` -->

![\Large \color{red}O\[n\]\color{green}+=\color{blue}I\[m,n\]](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7DO%5Bn%5D%5Ccolor%7Bgreen%7D%2B%3D%5Ccolor%7Bblue%7DI%5Bm%2Cn%5D)

In green, notice that the summation symbol is represented as `+=` in C++ Tile
code. Some portions of the notation do not perfectly correspond. Here's why:

- Summation notation includes a `m` subscript to indicate that `m` is the
  variable being summed over. Tile code implicitly sums over all valid indices
  (valid means not out of range for any tensor, and not failing any additional
  user-specified constraints as discussed in later examples).
- Tile must be explicitly given the shape of any new tensor created, done in
  this code by `TensorOutput(N)`. In this case we want `N` to match the size of
  the last dimension of `I`, which is specified by using `I.bind_dims(M, N)`.
  It is possible, however, to make this dimension of `O` larger or smaller,
  which would zero-pad or truncate `O` respectively.

  For example,

  ```c++
  auto O = TensorOutput(N + 1);
  ```

  would result in a `0` as the last element of `O` if we're still assuming `N`
  is the size of the last dimension of `I`.

- As is the case for all C++ statements, they must end with a semicolon.

### Max Over Axis

Taking the maximum over axis `0` looks very similar to taking the sum over axis
`0`. Just like a sum is represented in Tile with `+=`, a max is represented by
`>=`. Thus, the Tile code for max over axis `0` is just a single character
change from sum over axis `0`. Let's look at it as a Tile function:

```c++
Tensor max_over_axis(const Tensor& I) {
  TensorDim M, N;
  TensorIndex m, n;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(n) >= I(m, n);
  return O;
}
```

Again, this corresponds closely to mathematical notation:

<!-- 
```math
 \Large
 \textcolor{red}{O[n]}
 \textcolor{yellow}{=}
 \textcolor{green}{\max_m}
 \textcolor{cyan}{I[m, n]}
 ```

 ```math
 \Large
 \texttt{
   \textcolor{red}{O(n)}
   \textcolor{green}{>=}
   \textcolor{cyan}{I(m, n)};
 }
 ``` -->

![\Large \color{red}O\[n\]\color{black}= \color{green}\max_{m}{\color{blue}I\[m,n\]}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7DO%5Bn%5D%5Ccolor%7Bblack%7D%3D%20%5Ccolor%7Bgreen%7D%5Cmax_%7Bm%7D%7B%5Ccolor%7Bblue%7DI%5Bm%2Cn%5D%7D)

![\Large \color{red}O\[n\] \color{green} > = \color{blue}I\[m,n\]](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7DO%5Bn%5D%20%5Ccolor%7Bgreen%7D%20%3E%20%3D%20%5Ccolor%7Bblue%7DI%5Bm%2Cn%5D)

### Matrix Multiply

Next we'll consider matrix multiplication. Let's look at the mathematical
expression for the matrix multiplication `C = AB` written out in element-level
detail:

<!-- ```math
 \Large
 C[i, j] = \sum_{k} (A[i, k] \cdot B[k, j])
 ```  -->

![\Large C\[i, j\] = \sum_{k} (A\[i, k\] \cdot B\[k, j\])](https://render.githubusercontent.com/render/math?math=%5CLarge%20C%5Bi%2C%20j%5D%20%3D%20%5Csum_%7Bk%7D%20(A%5Bi%2C%20k%5D%20%5Ccdot%20B%5Bk%2C%20j%5D))

We can convert this to C++ Tile code using the same correspondence as the
previous example: The summation sign becomes plus-assignment, the summation
index is omitted, dimensions are given for the output tensor, and the statement
ends in a semicolon. Here's the result:

```c++
C(i, j) += A(i, k) * B(k, j);
```

To have correct dimensions, we need `I` to be the first dimension of `A` and `J`
the last dimension of `B`. Here's how this looks as part of a full Tile
function:

```c++
Tensor matmul(const Tensor& A, const Tensor& B) {
  TensorDim I, J, K;
  TensorIndex i, j, k;
  A.bind_dims(I, K);
  B.bind_dims(K, J);
  auto C = TensorOutput(I, J);
  C(i, j) += A(i, k) * B(k, j);
  return C;
}
```

Notice that we use `bind_dims` on inputs and we use `TensorOutput` on
outputs. Input dimensions can be repeated, which results in an error if the Tile
function is passed inputs whose corresponding dimensions don't all have the
specified size (for example `A.bind_dims(K, K)` would be constrained to a
square).

### Global Min

There is a min contraction `<=` analogous to the max contraction `>=`. For the
purposes of this example, however, let's use the formula `min(X) = -max(-X)`, to
compute the min. We do this by combining a max computation with _elementwise_
operations that perform the same operation (in this case negation) on every
element of a tensor. Elementwise operations generally cannot be performed on the
same line as contractions, so we write the global min function (for a 3D tensor)
as follows:

```c++
Tensor global_min(const Tensor& I) {
  TensorIndex i, j, k;
  auto Neg = -I;
  auto O_Neg = TensorOutput();
  O_Neg() >= Neg(i, j, k);
  auto O = -O_Neg;
  return O;
}
```

There are several novel pieces in this example. First, note that the elementwise
operations do not include dimensions. Dimensions are inferred from the inputs in
elementwise operations, and so are never specified in elementwise ops. `Neg` has
the same shape as `I`, and `O` has the same shape as `O_Neg`. When an
elementwise binary operation is performed, the output shape is determined using
[broadcasting semantics][].

Which brings us to the next novelty: we have our first example of a 0D tensor,
`O_Neg`. Tensors in Tile are allowed to have zero dimensions. In such a case the
tensor represents a scalar, i.e., a single value. In places where dimensions are
specified, you can indicate a 0-dimensional tensor by using `()` for the
dimensions, as in this example.

Notice that we are taking the max over all axes in a single operation.
Contractions implicitly aggregate over _all_ indices that write to the same
output location (in this case we aggregate over all values of `i`, `j`, and
`k`).

### Average

To compute the mean of a tensor, we need to sum the elements and divide by the
total number of elements summed. We can do this by taking advantage of the fact
that we can divide by a constant (including an input `TensorDim`) as an
elementwise operation. Thus, to take the mean over axis `0` of a 2D tensor, we
write:

```c++
Tensor avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum(y) += I(x, y);
  return Sum / X;
}
```

We can perform multiple elementwise operations on the same line, including
operations on constants and input dimensions. So, while it would be possible to
take a global mean of a 2D tensor in stages as so:

```
Tensor avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  PartialMean = Sum / X;
  return PartialMean / Y;
}
```

it is more straightforward to merge the elementwise operations:

```
Tensor avg(const Tensor& I) {
  TensorDim X, Y;
  TensorIndex x, y;
  I.bind_dims(X, Y);
  auto Sum = TensorOutput();
  Sum() += I(x, y);
  return Sum / (X * Y);
}
```

### Max Pool 1D

Next let's implement a size 2 stride 2 maxpool in Tile. This is the operation
that splits a tensor into groups of 2 and takes the larger element from each
group, yielding a tensor of half the original size. This is straightforward to
implement in straight C++:

```c++
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
```

`for` loops over tensor indices get translated into contractions when written in
Tile. The most direct (and, sadly, wrong) implementation in Tile is:

```c++
Tensor wrong_max_pool_1d(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  return O;
}
```

If you were to run this code, every entry of `O` would equal the global max of
`I`. We correctly determined that this was a maximization operation, and the
indices for `O` and `I` match those used in the straight C++ code, so what went wrong?

The problem with this Tile code is that there are too many "valid" indices. For
example, the case `i = 1`, `j = 3` means that `O[1]` checks `I[5]` as one of the
potential maximum values, even though `O[1]` is intended to be `max(I[2], I[3])`.
When we wrote the code with for loops, the inner loop restricted `j` to `0` or
`1`; in the Tile code, the compiler figured out the allowed values of `j` by
looking at the shapes of the tensors, and the only restriction that imposes on
`j` is that `j` must be an integer satisfying `0 <= 2 * i + j < N`.

When can use `if` statements in Tile to handle such situations:

```c++
Tensor max_pool_1d(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput(N / 2);
  O(i) >= I(2 * i + j);
  O.add_constraints({j<2});
  return O;
}
```

Something important to note here is that while we wrote `j < 2`, this constraint
actually means `0<= j < 2`. Constraints are always bounded below by `0`.
(Without a constraint, however, index variables may still be negative: the
original code included e.g. `i = 1`, `j = -1` as valid index pair.)

We determined the Tile code for this example by starting from imperative code,
but this Tile code is still very similar to mathematical notation, and we could
have started there instead:

<!-- ```math
 \Large
 \textcolor{red}{O[n]}
 \textcolor{yellow}{=}
 \textcolor{green}{\max}\textcolor{magenta}{_{0 \leq j < 2}}
 \textcolor{cyan}{I[2i + j]}
 ```

 ```math
 \Large
 \texttt{
   if (\textcolor{magenta}{j < 2}) \{{
     \textcolor{red}{O(n)}
     \textcolor{green}{>=}
     \textcolor{cyan}{I(2 * i + j)};
   \}}
 }
 ``` -->

![\Large \color{red}O\[n\] \color{black}= \color{green}\max_{\color{magenta}0 \ge j < 2}{\color{blue} I\[2i+j\]}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7DO%5Bn%5D%20%5Ccolor%7Bblack%7D%3D%20%5Ccolor%7Bgreen%7D%5Cmax_%7B%5Ccolor%7Bmagenta%7D0%20%5Cge%20j%20%3C%202%7D%7B%5Ccolor%7Bblue%7D%20I%5B2i%2Bj%5D%7D)

![\Large \color{black}if (\color{magenta}j < 2\color{black})  \{\color{red} O\\[n\\] \color{green} > = \color{blue} I (2*i+j) \color{black};\}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bblack%7Dif%20(%5Ccolor%7Bmagenta%7Dj%20%3C%202%5Ccolor%7Bblack%7D)%20%20%5C%7B%5Ccolor%7Bred%7D%20O%5C%5Bn%5C%5D%20%5Ccolor%7Bgreen%7D%20%3E%20%3D%20%5Ccolor%7Bblue%7D%20I%20(2*i%2Bj)%20%5Ccolor%7Bblack%7D%3B%5C%7D)

This Tile code handles odd values of `N` by rounding down the output tensor
size. You may instead want to round up the output tensor size and use a smaller
pool at the edge. This can be accomplished by simply adjusting the size of `O`:

```c++
Tensor max_pool_1d(const Tensor& I) {
  TensorDim N;
  TensorIndex i, j;
  I.bind_dims(N);
  auto O = TensorOutput((N + 1) / 2);
  if (j < 2) {
    O(i) >= I(2 * i + j);
  }
  return O;
}
```

No special handling is needed for the case `i = (N - 1) / 2`, `j = 1`; this is
out of range for `I` and so is ignored by Tile, which is exactly the intended
behavior.

### Valid Indices

When discussing contractions, we've mentioned that they accumulate over "all
valid indices". Hopefully the significance of this has been clear for the
specific examples we've looked at, but to write complex or novel code it helps
to have a precise understanding of what is meant by "valid indices".

First, index validity is determined for a full set of index variables: `j = 1`
is not valid or invalid as a standalone index value, but may be part of a valid
or invalid set of index variables. For example, in the code:

```c++
I.bind_dims(N);
auto O = TensorOutput((N + 1) / 2);
O(i) >= I(2 * i + j);
O.add_constraints({j<2});
}
```

with `N = 5`, the indices `i = 1, j = 1` are valid indices.
However, `i = 2, j = 1` are not valid indices for this operation, nor are `i = -1000, j = 1`.

A set of indices are _valid_ if and only if:

1. All the index variables are integers.
1. All the index expressions for every tensor are in range. Specifically, if the
   index variable values are plugged into every index expression, all the
   resulting indices are non-negative integers less than the appropriate
   dimension.
1. All the constraints are satisfied.
   Constraints always take the form `[index expression] < [constant expression]`
   (where `[index expression]` is a linear polynomial in the index
   variables and `[constant expression]` is a linear polynomial in the input
   dimensions), and they always implicitly include `0 <= [index expression]`.
   Therefore we could also state this requirement as "every constraint's index
   expression is non-negative and less than its specified upper bound".

### Skipping

The rule that all index variables must be integers allows us to "skip" certain
otherwise valid entries. For example, consider the Tile function:

```c++
Tensor skip(const Tensor& I) {
  TensorDim M, N;
  TensorIndex i, j;
  I.bind_dims(M, N);
  auto O = TensorOutput(N);
  O(2 * i) += I(2 * i, j);
  return O;
}
```

This operation only writes to even entries of `O`; while `i = 1/2, j = 1` does
yield valid index expressions (`O[1]` and `I[1, 1]`), using a fractional index
variable `i` makes these indices invalid. Note that some elements of `O` are
never written to. Any unwritten elements in the output of a contraction are
initialized to `0`.

### Cumulative Sum

Suppose we want to take the cumulative sum of a 1D tensor. That is, we want
`O[i]` to be the sum of all input entries `I[k]` where `k <= i`. In summation
notation, this is:

<!-- ```math
 \Large
 O[i] = \sum_{k \leq i} I[k]
 ``` -->

![\Large O\[i\] = \sum_{k \leq i} I\[k\]](https://render.githubusercontent.com/render/math?math=\%5CLarge%20O%5Bi%5D%20%3D%20%5Csum_%7Bk%20%5Cleq%20i%7D%20I%5Bk%5D)

However, we can't use `k <= i` as a constraint in Tile; all the index variables
must be gathered into a single index expression on one side of the inequality.
Thus, we rewrite this as `0 <= i - k`. Since the `0` bound is implicitly included
in all constraints, we just need to choose an upper bound large enough to never
be hit. From the dimensions of the tensors, we already know `i < N` and `0 <= k`,
and so `N` is an appropriate upper bound. The resulting Tile code is:

```c++
Tensor csum(const Tensor& I) {
  TensorDim N;
  TensorIndex i, k;
  I.bind_dims(N);
  auto O = TensorOutput(N);
  O(i) += I(k);
  O.add_constraints({i - k < N});
  return O;
}
```

Alternatively, we could write `k = i - j` for `j` non-negative as an alternative
way of forcing `k` to be no larger than `i`. Then in summation notation we have:

<!-- ```math
 \Large
 \textcolor{red}{O[i]}
 \textcolor{yellow}{=}
 \textcolor{green}{\sum}\textcolor{magenta}{_{0 \leq j}}
 \textcolor{cyan}{I[i - j]}
 ``` -->

 <!-- 
```math
 \Large
 \texttt{
   if (\textcolor{magenta}{j < N}) \{{
     \textcolor{red}{O(n)}
     \textcolor{green}{+=}
     \textcolor{cyan}{I(i - j)};
   \}}
 }
 ``` -->

![\Large  \color{red} O\[i\]  \color{black} =  \color{green} \sum _{\color{magenta}0  \leq  j}{\color{blue}I\[i - j\]}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%20%5Ccolor%7Bred%7D%20O%5Bi%5D%20%20%5Ccolor%7Bblack%7D%20%3D%20%20%5Ccolor%7Bgreen%7D%20%5Csum%20_%7B%5Ccolor%7Bmagenta%7D0%20%20%5Cleq%20%20j%7D%7B%5Ccolor%7Bblue%7DI%5Bi%20-%20j%5D%7D)

![\Large \text{if ( } \color{magenta} j < N \color{black} \text{)}  \{ \color{red}O\[n\] \color{green} += \color{blue} I( I - j ) \color{black};\}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ctext%7Bif%20(%20%7D%20%5Ccolor%7Bmagenta%7D%20j%20%3C%20N%20%5Ccolor%7Bblack%7D%20%5Ctext%7B)%7D%20%20%5C%7B%20%5Ccolor%7Bred%7DO%5Bn%5D%20%5Ccolor%7Bgreen%7D%20%2B%3D%20%5Ccolor%7Bblue%7D%20I(%20I%20-%20j%20)%20%5Ccolor%7Bblack%7D%3B%5C%7D)

### Convolution

Let's implement a 1D convolution with output size equal to input size. This is
implementing the Keras backend operation:

```python
K.conv1d(x, kernel, padding='valid')
```

Let's start with the mathematical formula for this operation:

<!-- ```math
 \Large
 O[n, x, c_o] = \sum_k \sum_{c_i}(I[n, x + k, c_i] \cdot K[k, c_i, c_o])
 ``` -->

![\Large O\[n, x, c_o\] = \sum_k \sum_{c_i}(I\[n, x + k, c_i\] \cdot K\[k, c_i, c_o\])](https://render.githubusercontent.com/render/math?math=%5CLarge%20O%5Bn%2C%20x%2C%20c_o%5D%20%3D%20%5Csum_k%20%5Csum_%7Bc_i%7D(I%5Bn%2C%20x%20%2B%20k%2C%20c_i%5D%20%5Ccdot%20K%5Bk%2C%20c_i%2C%20c_o%5D))

This is rather complicated, so let's walk through why this is the same
convolution formula we're used to in machine learning.

A convolution produces output for a specific batch element at a specific
location in a specific channel by taking a weighted sum of the input for that
same batch element at that same location _and a surrounding region_ over all
input channels. The weights are given by `K`, which depends on the output
channel, the input channel, and the displacement within the input region
relative to the reference location.

This generally matches the given formula: The output `O` is given as a sum of
elements from the input `I`, weighted by `K`. Looking at the meaning of the
index variables, we see that it matches exactly:

- `n` represents which element of the batch we're on.
- `ci` represents which input channel we're on.
- `co` represents which output channel we're on.
- `x` represents our spatial location, giving the location being written to in
  `O` and the smallest element read from in `I`.
- Finally, `k` represents the kernel offset, that is, how far (in the spatial
  dimension) the input element we're reading is from the lower bound of the
  kernel.

This formula directly translates to Tile, although note that `padding='valid'`
means that the spatial dimension of the output will be reduced by one less than
the kernel size relative to the spatial dimension of the input:

<!-- ```math
 \Large
 \textcolor{red}{O[n, x, c_o]}
 \textcolor{yellow}{=}
 \textcolor{green}{\sum_k \sum_{c_i}}
 \textcolor{cyan}{I[n, x + k, c_i]}
 \textcolor{orange}{\cdot}
 \textcolor{lightblue}{K[k, c_i, c_o]}
 ```
 ```math
 \Large
 \texttt{
   \textcolor{red}{O(n, x, co)}
   \textcolor{green}{+=}
   \textcolor{cyan}{I(n, x + k, ci)}
   \textcolor{orange}{*}
   \textcolor{lightblue}{K(k, ci, co)};
 }
 ``` -->

![\Large \color{red} O\[n, x, c_o\] \color{black} = \color{green} \sum_k \sum_{c_i} \color{blue} I\[n, x + k, c_i\] \color{orange} \cdot \color{lightblue}K\[k, c_i, c_o\]](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7D%20O%5Bn%2C%20x%2C%20c_o%5D%20%5Ccolor%7Bblack%7D%20%3D%20%5Ccolor%7Bgreen%7D%20%5Csum_k%20%5Csum_%7Bc_i%7D%20%5Ccolor%7Bblue%7D%20I%5Bn%2C%20x%20%2B%20k%2C%20c_i%5D%20%5Ccolor%7Borange%7D%20%5Ccdot%20%5Ccolor%7Blightblue%7DK%5Bk%2C%20c_i%2C%20c_o%5D)

![\Large \color{red} O(n, x, co)  \color{green} += \color{blue}I(n, x + k, ci) \color{orange} * \color{lightblue}K(k, ci, co)](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Ccolor%7Bred%7D%20O(n%2C%20x%2C%20co)%20%20%5Ccolor%7Bgreen%7D%20%2B%3D%20%5Ccolor%7Bblue%7DI(n%2C%20x%20%2B%20k%2C%20ci)%20%5Ccolor%7Borange%7D%20*%20%5Ccolor%7Blightblue%7DK(k%2C%20ci%2C%20co))

```c++
Tensor conv_1d(const Tensor& I, const Tensor& K) {
  TensorDim N, X, KX, CI, CO;
  TensorIndex n, x, k, ci, co;
  I.bind_dims(N, X, CI);
  K.bind_dims(KX, CI, CO);
  auto O = TensorOutput(N, X - KX + 1, CO);
  O(n, x, co) += I(n, x + k, ci) * K(k, ci, co);
  return O;
}
```

### Dilated 2D Convolution

We can tweak this general formula for a convolution to add various features,
such as different strides, changing the padding, performing the convolution
depthwise, etc. For this example, we will implement a dilated 2D convolution
with dilation rate (2, 3). Specfically, we'll implement the Keras backend
function:

```python
K.conv2d(x, kernel, padding='valid', dilation_rate=(2, 3))
```

The formula for this is very similar to the previous convolution; we just have
an additional spatial dimension for each tensor, and the kernel offset index
variables are multiplied by dilation scaling factors when used to determine
indices for `I`:

<!-- ```math
 \Large
 O[n, x, y, c_o] = \sum_{k_x} \sum_{k_y} \sum_{c_i}
 I[n, x + 2k_x, y + 3k_y, c_i] *
 K[k_x, k_y, c_i, c_o]
 ``` -->

![\Large O\[n, x, y, c_o\] = \sum_{k_x} \sum_{k_y} \sum_{c_i} I\[n, x + 2k_x, y + 3k_y, c_i\] * K\[k_x, k_y, c_i, c_o\]](https://render.githubusercontent.com/render/math?math=%5CLarge%20O%5Bn%2C%20x%2C%20y%2C%20c_o%5D%20%3D%20%5Csum_%7Bk_x%7D%20%5Csum_%7Bk_y%7D%20%5Csum_%7Bc_i%7D%20I%5Bn%2C%20x%20%2B%202k_x%2C%20y%20%2B%203k_y%2C%20c_i%5D%20*%20K%5Bk_x%2C%20k_y%2C%20c_i%2C%20c_o%5D)

The effective size for a dilated kernel with kernel size `K` and dilation rate
`d` is `d * (K - 1) + 1`, and so to achieve `'valid'` padding for this
convolution, the x dimension must be reduced by `2 * (KX - 1)` and the y
dimension must be reduced by `3 * (KY - 1)`, where `KX` and `KY` are the x and y
dimensions of the kernel respectively. The rest of the Tile code corresponds
directly to the formula, and so we get:

```c++
Tensor conv_2d(const Tensor& I, const Tensor& K) {
  TensorDim N, X, Y, KX, KY, CI, CO;
  TensorIndex n, x, y, kx, ky, ci, co;
  I.bind_dims(N, X, Y, CI);
  K.bind_dims(KX, KY, CI, CO);
  auto O = TensorOutput(N, X - 2 * (KX - 1), Y - 3 * (KY - 1), CO);
  O(n, x, y, co) += I(n, x + 2 * kx, y + 3 * ky, ci) * K(kx, ky, ci, co);
  return O;
}
```

### Complex Convolution

This final example demonstrates a strided dilated padded grouped convolution.

<!-- ```math
 \Large
 \begin{aligned}
 O&[n, x_0, x_1, g, c_{o, g}] \\
 &= \sum_{k_0, k_1, c_{i, g}}
 (
   I[n, s_0 x_0 + d_0 k_0 - P_0, s_1 x_1 + d_1 k_1 - P_1, c_{i, g}] *
   K[k_0, k_1, g, c_{i, g}, c_{o, g}]
 )
 \end{aligned}
 ``` -->

![\Large \begin{aligned} O&\[n, x_0, x_1, g, c_{o, g}\] \\ &= \sum_{k_0, k_1, c_{i, g}} (   I\[n, s_0 x_0 + d_0 k_0 - P_0, s_1 x_1 + d_1 k_1 - P_1, c_{i, g}\] *   K\[k_0, k_1, g, c_{i, g}, c_{o, g}\] ) \end{aligned}](https://render.githubusercontent.com/render/math?math=%5CLarge%20%5Cbegin%7Baligned%7D%20O%26%5Bn%2C%20x_0%2C%20x_1%2C%20g%2C%20c_%7Bo%2C%20g%7D%5D%20%5C%5C%20%26%3D%20%5Csum_%7Bk_0%2C%20k_1%2C%20c_%7Bi%2C%20g%7D%7D%20(%20%20%20I%5Bn%2C%20s_0%20x_0%20%2B%20d_0%20k_0%20-%20P_0%2C%20s_1%20x_1%20%2B%20d_1%20k_1%20-%20P_1%2C%20c_%7Bi%2C%20g%7D%5D%20*%20%20%20K%5Bk_0%2C%20k_1%2C%20g%2C%20c_%7Bi%2C%20g%7D%2C%20c_%7Bo%2C%20g%7D%5D%20)%20%5Cend%7Baligned%7D)

where _`s`_ gives the stride coefficients, _`d`_ gives the dilation
coefficients, and _`P`_ gives the padding offsets.

```c++
Tensor complex_conv_2d(
    const Tensor& I,
    const Tensor& K,
    const std::vector<size_t>& s,  // stride coeffs
    const std::vector<size_t>& d   // dilation coeffs
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
    Y[i] = (X[i] + s[i] - 1) / s[i];
  }
  // Compute the effective kernel size after dilation
  std::vector<TensorDim> EK(2);
  for (size_t i = 0; i < EK.size(); ++i) {
    EK[i] = d[i] * (K[i] - 1) + 1;
  }
  // Compute the padding offset
  std::vector<TensorDim> P(2);
  for (size_t i = 0; i < P.size(); ++i) {
    P[i] = ((Y[i] - 1) * s[i] + EK[i] - X[i]) / 2;
  }
  // Specify the output size
  auto O = TensorOutput(N, Y0, Y1, G, GCO);
  // Compute the convolution
  O(n, x[0], x[1], g, gco) +=
      I(n, s[0]*x[0] + d[0]*k[0] - P[0], s[1]*x[1] + d[1]*k[1] - P[1], g, gci) *
      K(k0, k1, g, gci, gco);
  return O;
}
```

## Reference

### Contractions

There are five _aggregation_ operations:

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
two tensors used inside the contraction, you also choose a _combination_
operation to determine how their values are combined. The only combination
operations that are currently well-supported are multiplication (`*`) and
addition (`+`).

Contractions aggregate over all sets of _valid indices_. A set of indices is
valid for a contraction if and only if:

- All index variables are integers
- All index expressions used in tensors are within bounds
- All user-specified constraints are satisfied

### Elementwise Operations

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

### Types

- `Tensor`: Multidimensional arrays of a fixed shape. The scope of a tensor is
  the entire function. By convention, tensors begin with a capital letter.
- `TensorDim`: Positive integers initially passed to a function as sizes of
  input tensors. The scope of a dimension is the entire function. By convention,
  dimensions begin with a capital letter.
- `TensorIndex`: Symbolic integers used in contractions to directly index a
  tensor or as part of a formula to compute a tensor index. The scope of an
  index is a single operation. By convention, indices begin with a lower case
  letter.

[broadcasting semantics]: https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
