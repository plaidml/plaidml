# Op Library Design Decisions

This documents why we made various design decisions in creating this operation library.

## Tensor Layout

Operations like convolution and pool need to know the semantic layout of the tensor, i.e., what's the batch dimension, what're the spatial dimensions, what's the channel dimension (or dimensions, for convolution kernels or if e.g. you blocked the channel dim and separated the low and high order portions of the channel dim). We have chosen to provide this layout information as a parameter to operations that need to know tensor layouts.

We decided to do this instead of setting a global layout that would apply to every operations because we thought users might sometimes want a mix of layouts throughout their network.

We decided to provide layout information with operations instead of with tensors because it is not possible to infer the layout of outputs for all operations (e.g., there is no way to infer the layout of the output of a reshape operation from the layout of the input). Thus, for at least some operations, layout would still need to be specified on the operation, not just the input tensors. Moreover, the operations that would need layout information in this model (e.g. reshape) are not operations where users are historically expected to provide layout information when they are using a layout other than the framework default. Instead, users have historically been expected to provide layout information on convolution or pooling operations when they are using a layout other than the framework default. By matching this historical usage in PlaidML, we intend to make it easier for current models to be used with PlaidML.

All that said, we believe tensor layout is a property of tensors more than it is a property of operations on tensors. With tensor compilers like PlaidML, it should not be necessary to store layout information on operations to ensure optimized operations. Therefore, we believe it would be more natural to store layout information on the tensors, and that this would not cause performance degradations. Moreover, for the more frequent convolution and pooling operations, output tensor layouts could be infered automatically. Tensor layouts would only need to be provided with operations like reshape. This has the additional benefit of providing layout information on operations which change the semantics of the tensor's layout. If we did not have the historical constraints discussed above, we believe providing layout information on tensors rather than (most) operations would be the preferred solution.

## Strings for Enumerated Input Parameters

Parameters representing one of a choice of options are passed as strings. (These show up as e.g. the autopadding mode for convolutions/pooling.) These strings are then converted to an appropriate enum in the op library. As part of this process, we verify that the string contains expected text, and we throw if we receive something surprising.

We made this decision because passing through the ffi layer would require either providing this parameters as strings or casting the enum representing the parameter to an integer, then casting it back. We believe passing the parameter as a string makes it harder to accidentally pass the wrong value. We convert the parameters to enums because they semantically represent enum data, and also because the conversion step is an opportunity to check for unexpected parameter values.