# Adding Operations

Sometimes, when implementing a cutting-edge or unusual network, the operations you need are not available in the frontend you're using or in the PlaidML common operations library. This tutorial will show you how to write a new operation in PlaidML.

Accompanying the instructions will be a demonstration of how these techniques can be used to write the Keras backend function ``categorical_crossentropy``.

## Write a Simple Case in Tile
PlaidML performs machine learning operations by executing code written in the
Tile language. Tile is useful for being relatively easy to translate into both
optimized GPU code and also mathematical notation, but it is not well-suited to
generalizing similar operations. For instance, the same operation applied to a
tensor with a different number of dimensions must be written with different Tile
code.

It is often easiest to first write a simple case in Tile, for which PlaidML
serves as a thin wrapper. Tile code is covered in
[writing_tile_code](writing_tile_code.md), so we'll keep the explanation of the
Tile code for categorical crossentropy to a minimum.

As our simple case for ``categorical_crossentropy``, we'll assume that the
``target`` and ``output`` tensors are 2 dimensional. In this case, the
mathematical expression of categorical crossentropy is

<div align=center><img src="images/math-cat-xent-raw.png" height="60"
alt="R[x] = sum_y -ln(O[x, y]) * T[x, y]"></div><br>

A key feature of this formula is the summation over the index ``y``. In Tile,
such operations cannot typically be performed on the same line as elementwise
operations, so we will split this formula into three parts:

<div align=center><img src="images/math-cat-xent-split.png" height="120pt"
alt="LO[x, y] = ln(O[x, y]), Temp[x] = sum_y LO[x, y] * T[x, y], R[x] = -
Temp[x]"></div><br>

These formulas correspond quite directly with the Tile code for this operation:

```tile
function (T[X, Y], O[X, Y]) -> (R) {
    LO = log(O);
    Temp[x: X] = +(LO[x, y] * T[x, y])
    R = -Temp;
}
```

## Wrap the Tile Code

Keras can't interpret Tile code, but the PlaidML backend can. To turn Tile code
into a PlaidML operation, create a new subclass of :doc:`api/
plaidml.tile.Operation` and override its ``__init__`` function. During this
initialization you will need to call the superclass's ``__init__`` with the
following data:
- The Tile code to be executed.
- A list containing input variable data. Each input is represented as a two
element tuple in the list. The first element is the name of the input as a
string; the second is the tensor to be input. The names must match the names
used in the Tile code.
- A list containing output variable data. Each output is represented as a two
element tuple in the list. The first element is the name of the output as a
string; the second is the [api/plaidml.tile.Shape](api/plaidml.tile.Shape.rst)
of the output.

Note that a [api/plaidml.tile.Shape](api/plaidml.tile.Shape.rst) consists of a
[api/plaidml.DType](api/plaidml.DType.rst) and a tuple of dimensions; it is often useful to construct an output's shape by copying the dtype of an input shape and constructing some manipulation of the dimensions of one or more inputs. In general, a [api/plaidml.tile.Shape](api/plaidml.tile.Shape.rst)'s dimensions can be integers or [api/plaidml.Var](api/plaidml.Var.rst)s, the latter case typically constructed by some manipulation of dimensions of other tensors.

We can now write the Python code for this case of categorical crossentropy! It won't work if we get tensors of the wrong dimension, but in the 2D ``from_logits=False`` case, this is sufficient:
```python
class CategoricalCrossentropy(plaidml.tile.Operation):
  def __init__(self, target, output):
      code = """
              function (O[X, Y], T[X, Y]) -> (R) {
                  LO = Log(O);
                  Temp[x: X] = +(LO[x, y] * T[x, y]);
                  R = -Temp;
              }"""
      super(CategoricalCrossentropy, self).__init__(code, [('O', output), ('T', target)],
              [('R', plaidml.tile.Shape(output.shape.dtype, output.shape.dims[:-1]))])
```

Note one parameter that isn't needed: Tile code for the gradient. PlaidML
includes an autodiff utility for Tile that Keras can invoke to produce and
evaluate Tile code for the gradient of any Tile function. You only need to write
forward-pass operations; Keras and PlaidML will work out their gradients
automatically. Even for frontends without autodiff support you do not add
gradient information here; instead you may directly use
[api/plaidml.op.Gradients](api/plaidml.op.Gradients.rst) when constructing your 
model.

## Generalize

Having written Tile code for one case (crossentropy in 2D tensors) we generalize to all the cases we want the function to handle (crossentropy in
arbitrary-dimensional tensors; also accept logits). Tile is not designed for
this sort of generalization, so we change what Tile code we write using
substitution and string manipulation.

For categorical crossentropy, we change the number of dimensions by adding (or
removing) dimension sizes ``X`` and corresponding indices ``x``. We want a
number of each equal to ``output.shape.ndims - 1``, so we write the following:
```python
fixed_dims = ','.join('X{}'.format(i) for i in range(output.shape.ndims - 1))
fixed_idxs = ','.join('x{}'.format(i) for i in range(output.shape.ndims - 1))
```

We substitute these into the Tile code using the Python string ``format``
function:
```python
code = """
    function (O[{fixed_dims},Y], T[{fixed_dims},Y]) -> (R) {{
        LO = log(O);
        Temp[{fixed_idxs}:{fixed_dims}] = +(T[{fixed_idxs},y] * LO[{fixed_idxs},y]);
        R = -Temp;
    }}""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
```

We could handle ``from_logits`` by manipulating the Tile code in a similar way.
However, that case merely requires performing a softmax first, and softmax is
already defined in the common op library! So we instead add python code:
```python
if from_logits:
    output = plaidml.op.softmax(output, axis=output.shape.ndims - 1)
```

Putting it all together, we have:
```python
class CategoricalCrossentropy(plaidml.tile.Operation):
    def __init__(self, target, output, from_logits=False):
        if from_logits:
            output = plaidml.op.softmax(output, axis=output.shape.ndims - 1)
        fixed_dims == ", ".join(["X{}.format(i) for i in range(target.ndim - 1)])
        fixed_idxs == ", ".join(["x{}.format(i) for i in range(target.ndim - 1)])
        f = """function (T[{fixed_dims}, Y], O[{fixed_dims}, Y]) -> (R) {{
                   LO = Log(O);
                   Temp[{fixed_idxs}: {fixed_dims}] = +(LO[{fixed_idxs}, y] * T[{fixed_idxs}, y]);
                   R = -Temp;
               }}""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
        super(CategoricalCrossentropy, self).__init__(code, [('O', output), ('T', target)],
                [('R', plaidml.tile.Shape(output.shape.dtype, output.shape.dims[:-1]))])
```

## Add Tests, Handle Edge Cases

If you were to test the above code, you would find that it worked great ...
except if you passed it 1D tensors. That's mostly fine (especially in Keras
where nearly everything is batched), but "mostly fine" will come back to haunt
you, so you should handle that edge case (this is left as an exercise for the
reader). If you compare to the code we actually use for this in
``plaidml.keras.backend.CategoricalCrossentropy``, you'll see that we also
preprocess ``output`` if ``from_logits`` is False but the input is not directly
from softmax. This won't change ``output`` if it comes from a softmax, but it
will prevent domain errors from log that can occur if someone (improperly)
passes the function a non-softmaxed tensor.

## Wrap with Frontend Code

For ``categorical_crossentropy`` in Keras, we just need to assign the standard
Keras backend API function name ``categorical_crossentropy``:

```
categorical_crossentropy = CategoricalCrossentropy.function
```

This is a standard Keras backend function and Keras will use it where it needs
it. It is also polite to see whether other frontends use the operation you
added. If so, you can add the class to [api/plaidml.op](api/plaidml.op.rst)
where it can be referenced by PlaidML backend code for any frontends that need
it.

If you are creating a novel operation, you may want to wrap this backend
function in a higher-level frontend object. You will need to look at your
frontend's documentation for details on how to do this; e.g., for Keras see
[Writing your own Keras layers]. Depending on your purpose in adding the
operation, this may not be necessary: you can also use your custom operation
directly (typically by calling the ``function`` member function of your
operation, which is included as part of the [api/plaidml.tile.Operation]
class).

[Writing your own Keras layers]:https://keras.io/layers/writing-your-own-keras-layers
[api/plaidml.tile.Operation]:api/plaidml.tile.Operation.rst
