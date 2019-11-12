// Copyright 2019, Intel Corporation

include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"

//===----------------------------------------------------------------------===//
// PML op dialect definition
//===----------------------------------------------------------------------===//

def PML_Op_Dialect : Dialect {
   let name = "pml_op";
   let cppNamespace = "pmlc::dialect::pml_op";
}

//===----------------------------------------------------------------------===//
// PML op base class.
//===----------------------------------------------------------------------===//

class PML_Op<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
     Op<PML_Op_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// PML types
//===----------------------------------------------------------------------===//

// TODO(dgkutnic): Should we create a custom PML_Value type to mimic edsl::Value? I'm leaning towards using the base types for now...

//===----------------------------------------------------------------------===//
// PML op definitions.
//===----------------------------------------------------------------------===//
def PML_abs : PML_Op<"abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "PlaidML absolute value operation";

  let description = [{
description: 
    Takes a single tensor as an input, and returns the absolute value of that tensor as an output.
usage:
    y = abs(x)
required inputs:
    x                     the input tensor for which the absolute value will be computed
optional inputs:
    None
outputs:
    y                     contains the absolute value of the input tensor x
  }];

  let arguments = (ins AnyTensor:$x);

  let results = (outs AnyTensor:$y);
}

def PML_conv : PML_Op<"conv", [NoSideEffect]> {
  let summary = "PlaidML convolution operation";

  let description = [{
description:
    Placeholder
  }];

  let arguments = (
      ins AnyTensor:$x,
      I32ArrayAttr:$strides,
      I32ArrayAttr:$dilations,
      I32ArrayAttr:$data_dilations,
      I32ArrayAttr:$filter_shape,
      APIntAttr:$groups,
      StrAttr:$autopad_mode,
      I32ArrayAttr:$manual_padding,
      StrAttr:$filter_layout,
      StrAttr:$group_layout,
      BoolAttr:$winograd_allowed,
      StrAttr:$name,
      StrAttr:$autogroup_mode,
      StrAttr:$deriv_mode,
      I32ArrayAttr:$result_shape
  );

  let results = (outs AnyTensor:$y);
}
