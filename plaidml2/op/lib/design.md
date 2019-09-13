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

## Grouped Convolution: Autogrouping & Group Layout

For grouped convolutions, the input and output tensors have the same shapes as in classical convolutions, but the filter tensor (a.k.a. weights or kernel tensor) will have a different shape. There are multiple ways of laying out these dimensions. We specify this dimension layout as the `GroupLayout`, which is distinct from the `TensorLayout` that must also be specified for the filter tensor. We did these separately because the main `GroupLayout`s include the group as part of the input channels or output channels dimension on the filter tensor (and thus do not change what dimensions are in the tensor layout); we did not want duplicate each of the filter `TensorLayout`s two additional times to indicate which dimension tracked the group, especially as we did not find satisfactory names for such layouts. For `GroupLayout::SEPARATE`, where a separate group dimension is added to the filter tensor, we did add corresponding `TensorLayout`s indicating where this group dimension was laid out relative to the other dimensions.

We also want users to be able to either manually specify the number of groups or infer the number of groups from the input & filter tensor shapes. We call this inference of the number of groups "autogrouping". Autogrouping is only available with certain `GroupLayout`s: The `SEPARATE` and `IN_K` layouts allow autogrouping, but the number of groups cannot be infered when using the `IN_C` layout (unless output shape information is also provided, a case we did not set up autogrouping for as we considered it too niche). The AutogroupMode specifies what autogrouping to use, if any:
 * `UNGROUPED` if the convolution is not grouped
 * `EXPLICIT` if the number of groups will be provided by the user
 * `AUTO` for autogrouping as described above
 * `DEPTHWISE` for a limited autogrouping where the number of groups is equal to the number of input channels: this is to allow for depthwise convolutions with autogrouping while using the `IN_C` layout, a feature most notably useful for Keras.

The function `normalize_grouping_strategy` is used to ensure that the `GroupLayout`, the filter `TensorLayout`, and the user provided number of groups are mutually consistent; see comments in that function for details.

## Reshape as both a library op and an EDSL builtin

Reshape is available in two places: both in the op library at `op/lib/ops.cc:reshape` and in the EDSL at `edsl/edsl.h:reshape`. These have slightly different functionality:
 * The op library reshape includes auto-sizing features, where dimensions can be set to "whatever the corresponding dimension of the input is" or to "whatever size is necessary to make the flattened size of the reshaped tensor match the flattened size of the original tensor"
 * The EDSL reshape will reshape to a fully specified size. That is, each dimension is either an integer or a symbolic `TensorDim`; neither of the auto-sizing features described above is available.

The broader functionality of the op library reshape would be a bit messy to implement at the EDSL level. At the same time, having a more limited EDSL reshape gives us a reshape operation that can be easily translated to Stripe.
