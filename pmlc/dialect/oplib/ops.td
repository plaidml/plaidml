// Copyright 2019, Intel Corporation

include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// PML op dialect definition
//===----------------------------------------------------------------------===//

def PML_Op_Dialect : Dialect {
   let name = "pml_op";
   let cppNamespace = "pmlc::dialect::op";
}

//===----------------------------------------------------------------------===//
// PML op base class.
//===----------------------------------------------------------------------===//

class PML_Op<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
     Op<PML_Op_Dialect, mnemonic, traits> {}

//===----------------------------------------------------------------------===//
// PML types
//===----------------------------------------------------------------------===//

// enum attributes
def PML_NONE                : I32EnumAttrCase<"NONE", 0>;
def PML_EXPLICIT            : I32EnumAttrCase<"EXPLICIT", 0>;
// autogroup modes
def PML_AUTOGROUP_UNGROUPED : I32EnumAttrCase<"UNGROUPED", 1>;
def PML_AUTOGROUP_AUTO      : I32EnumAttrCase<"AUTO", 2>;
def PML_AUTOGROUP_DEPTHWISE : I32EnumAttrCase<"DEPTHWISE", 3>;
// autopad modes
def PML_AUTOPAD_NOTSET      : I32EnumAttrCase<"NOTSET", 0>;
def PML_AUTOPAD_SAME_LOWER  : I32EnumAttrCase<"SAME_LOWER", 1>;
def PML_AUTOPAD_SAME_UPPER  : I32EnumAttrCase<"SAME_UPPER", 2>;
def PML_AUTOPAD_VALID       : I32EnumAttrCase<"VALID", 3>;
// deriv modes
def PML_DERIV_DATA          : I32EnumAttrCase<"DATA", 1>;
def PML_DERIV_FILTER        : I32EnumAttrCase<"FILTER", 2>;
// group layouts
def PML_LAYOUT_SEPARATE     : I32EnumAttrCase<"SEPARATE", 1>;
def PML_LAYOUT_IN_C         : I32EnumAttrCase<"IN_C", 2>;
def PML_LAYOUT_IN_K         : I32EnumAttrCase<"IN_K", 3>;
// tensor layouts
def PML_LAYOUT_NXC          : I32EnumAttrCase<"NXC", 1>; 
def PML_LAYOUT_NCX          : I32EnumAttrCase<"NCX", 2>;
def PML_LAYOUT_KCX          : I32EnumAttrCase<"KCX", 3>;
def PML_LAYOUT_XCK          : I32EnumAttrCase<"XCK", 4>;
def PML_LAYOUT_GKCX         : I32EnumAttrCase<"GKCX", 5>;
def PML_LAYOUT_XGCK         : I32EnumAttrCase<"XGCK", 6>;

def PML_AutogroupModeAttr :
    I32EnumAttr<"AutogroupMode", "plaidml supported autogroup modes", [
        PML_EXPLICIT,            // Group size explicitly specified, > 1
        PML_AUTOGROUP_UNGROUPED, // Group size explicitly 1
        PML_AUTOGROUP_AUTO,      // Group size determined from shapes of I and F
        PML_AUTOGROUP_DEPTHWISE, // for channelized convolutions (i.e. where G = CI)
    ]> {
  let cppNamespace = "pmlc::dialect::op";
}

def PML_AutopadModeAttr :
    I32EnumAttr<"AutopadMode", "plaidml supported autopad modes", [
        PML_NONE,
        PML_AUTOPAD_NOTSET,      // NOTSET = NONE
        PML_EXPLICIT,            // EXPLICIT = NONE
        PML_AUTOPAD_SAME_LOWER,
        PML_AUTOPAD_SAME_UPPER,
        PML_AUTOPAD_VALID,
    ]> {
  let cppNamespace = "pmlc::dialect::op";
}

def PML_DerivModeAttr :
    I32EnumAttr<"DerivMode", "plaidml supported derivative computation modes", [
        PML_NONE,                // Forward pass
        PML_DERIV_DATA,          // Computing derivative of input data (or equivalently a transposed conv)
        PML_DERIV_FILTER,        // Computing derivative of filters
    ]> {
  let cppNamespace = "pmlc::dialect::op";
}

def PML_GroupLayoutAttr :
    I32EnumAttr<"GroupLayout", "plaidml supported group layouts", [
        PML_NONE,                // Not grouped
        PML_LAYOUT_SEPARATE,     // Group given as a separate dimension
        PML_LAYOUT_IN_C,         // Group included in the input channels dimension
        PML_LAYOUT_IN_K,         // Group included in the output channels
    ]> {
  let cppNamespace = "pmlc::dialect::op";
}

def PML_TensorLayoutAttr :
    I32EnumAttr<"TensorLayout", "plaidml supported tensor layouts", [
        PML_LAYOUT_NXC, 
        PML_LAYOUT_NCX, 
        PML_LAYOUT_KCX, 
        PML_LAYOUT_XCK, 
        PML_LAYOUT_GKCX, 
        PML_LAYOUT_XGCK,
    ]> {
  let cppNamespace = "pmlc::dialect::op";
}

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

  let arguments = (ins AnyTensor:$_x);

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
      AnyTensor:$filter,
      I32ArrayAttr:$strides,
      I32ArrayAttr:$dilations,
      I32ArrayAttr:$data_dilations,
      I32ArrayAttr:$kernel_shape,
      APIntAttr:$groups,
      PML_AutopadModeAttr:$autopad_mode,
      I32ArrayAttr:$manual_padding,
      PML_TensorLayoutAttr:$input_layout,
      PML_TensorLayoutAttr:$filter_layout,
      PML_GroupLayoutAttr:$group_layout,
      BoolAttr:$winograd_allowed,
      StrAttr:$name,
      PML_AutogroupModeAttr:$autogroup_mode,
      PML_DerivModeAttr:$deriv_mode,
      I32ArrayAttr:$result_shape
  );

  let results = (outs AnyTensor:$y);
}
