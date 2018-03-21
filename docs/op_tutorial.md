# Tile Op Tutorial
Sometimes, when implementing a cutting-edge or unusual network, the functions you need are not available in Keras. This tutorial will show you how to write a new operation in the PlaidML backend to Keras.

Accompanying the instructions will be a demonstration of how these techniques can be used to write the Keras backend function `categorical_crossentropy`.

## Write a Simple Case in Tile
Plaid performs machine learning operations by executing code written in the Tile language. Tile is useful for being relatively easy to translate into both optimized GPU code and also mathematical notation, but it is not well-suited to generalizing similar operations. For instance, the same operation applied to a tensor with a different number of dimensions must be written with different Tile code.

It is often easiest to first write a simple case in Tile, for which PlaidML and Keras serve as a thin wrapper. Tile code is covered in [[another tutorial|Tile-Tutorial]], so we'll keep the explanation of the Tile code for categorical crossentropy to a minimum.

As our simple case for `categorical_crossentropy`, we'll assume that the `target` and `output` tensors are 2 dimensional. In this case, the mathematical expression of categorical crossentropy is

<a href="url"><img src="docs/math-cat-xent-raw.png" height="60pt"></a>

A key feature of this formula is the summation over the index `y`. In Tile, such operations cannot typically be performed on the same line as elementwise operations, so we will split this formula into three parts:

<a href="url"><img src="docs/math-cat-xent-split.png" height="120pt"></a>

These formulas correspond quite directly with the Tile code for this operation.
```
function (T[X, Y], O[X, Y]) -> (R) {
	LO = log(O);
	Temp[x: X] = +(LO[x, y] * T[x, y])
	R = -Temp;
}
```

## Wrap the Tile Code
Keras can't interpret Tile code, but the PlaidML backend can. The class `_Op` (in our backend code `plaidml/keras/backend.py`) can be initialized with Tile code and will produce an object Keras can interact with as a Keras tensor. It requires several additional parameters as well:
 - **ident**: A string giving the name of the operation. Appears in `__repr__` and `__str__`.
 - **dtype**: The data type of the result tensor, e.g. `'float32'`.
 - **shape**: The Keras shape of the result tensor. A tuple of positive integers and/or Nones.
 - **code**: Tile code for the operation, given as a string.
 - **inputs**: An ordered dictionary. The key is the name of the input as a string; the value is the tensor to be input. Be sure the name of the first (and second, etc) input here is the same as the name of the first input in the Tile code.
 - **outputs**: A list of strings giving the names of the output(s). Please ensure the name(s) (and, if applicable, order) of the output(s) match(es) the Tile code.

We can now write the Python code for this case of categorical crossentropy! It won't work if we get tensors of the wrong dimension, but in the 2D `from_logits=False` case, this is sufficient:
```
def categorical_crossentropy(target, output):
	f = """function (T[X, Y], O[X, Y]) -> (R) {
	           LO = Log(O);
	           Temp[x: X] = +(LO[x, y] * T[x, y]);
	           R = -Temp;
	       }"""
	return _Op('cat_xentropy', output.dtype, output.shape[:-1], f,
	           OrderedDict([('T', target), ('O', output)]), ['R'])
```

Note one parameter that isn't needed: Tile code for the gradient. PlaidML includes an autodiff utility for Tile that Keras can invoke to produce and evaluate Tile code for the gradient of any Tile function. You only need to write forward-pass operations; Keras and PlaidML will work out their gradients automatically.

## Generalize
Having written Tile code for one case (crossentropy in 2D tensors) we generalize to all the cases we want the function to handle (crossentropy in arbitrary-dimensional tensors; also accept logits). Tile is not designed for this sort of generalization, so we change what Tile code we write using substitution and string manipulation.

For categorical crossentropy, we change the number of dimensions by adding (or removing) dimension sizes `X` and corresponding indices `x`. We want a number of each equal to `target.ndim - 1`, so we write the following:
```
fixed_dims == ", ".join(["X{}.format(i) for i in range(target.ndim - 1)])
fixed_idxs == ", ".join(["x{}.format(i) for i in range(target.ndim - 1)])
```
We substitute these into the Tile code using the Python string `format` function:
```
f = """function (T[{fixed_dims}, Y], O[{fixed_dims}, Y]) -> (R) {
           LO = Log(O);
           Temp[{fixed_idxs}: {fixed_dims}] = +(LO[{fixed_idxs}, y] * T[{fixed_idxs}, y]);
           R = -Temp;
       }""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
```

We could handle `from_logits` by manipulating the Tile code in a similar way. However, that case merely requires performing a softmax first, and softmax is already defined in the backend! So we instead add python code
```
if from_logits:
	output = softmax(output)
```

Putting it all together, we have
```
def categorical_crossentropy(target, output):
	if from_logits:
		output = softmax(output)
	fixed_dims == ", ".join(["X{}.format(i) for i in range(target.ndim - 1)])
	fixed_idxs == ", ".join(["x{}.format(i) for i in range(target.ndim - 1)])
	f = """function (T[{fixed_dims}, Y], O[{fixed_dims}, Y]) -> (R) {
	           LO = Log(O);
	           Temp[{fixed_idxs}: {fixed_dims}] = +(LO[{fixed_idxs}, y] * T[{fixed_idxs}, y]);
	           R = -Temp;
	       }""".format(fixed_dims=fixed_dims, fixed_idxs=fixed_idxs)
	return _Op('cat_xentropy', output.dtype, output.shape[:-1], f,
	           OrderedDict([('T', target), ('O', output)]), ['R'])
```

## Add Tests, Handle Edge Cases
If you were to test the above code, you would find that it worked great ... except if you passed it 1D tensors. That's mostly fine (especially in Keras where nearly everything is batched), but "mostly fine" will come back to haunt you, so you should handle that edge case (this is left as an exercise for the reader). If you compare to the [backend code](https://github.com/plaidml/plaidml/blob/master/plaidml/keras/backend.py) we actually use for this, you'll see that we also preprocess `output` if `from_logits` is False but the input is not directly from softmax. This won't change `output` if it is formatted like the result of a softmax, but it will prevent domain errors from log that can occur if someone (improperly) passes the function a non-softmaxed tensor.

## Wrap with Keras Code
For `categorical_crossentropy`, we're done: this is a standard Keras backend function and Keras will use it where it needs it. If you are creating a novel operation, however, you may want to wrap this backend function in a higher-level Keras object. For details on how to do this, see [the Keras documentation](https://keras.io/layers/writing-your-own-keras-layers/).

For some operations this is unnecessary. You can always call your custom backend function directly among your other Keras code, just like you can call a TensorFlow function directly if you're using the TensorFlow backend.