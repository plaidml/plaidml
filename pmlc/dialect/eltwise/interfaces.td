// Copyright 2019, Intel Corporation

#ifndef __PML_ELTWISE_INTERFACES__
#define __PML_ELTWISE_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def EltwiseOpInterface : OpInterface<"EltwiseOp"> {
  let description = "<TODO>";
  let methods = [
    InterfaceMethod<
      "Build a standard op from eltwise on in conversion", 
      "Value*", "buildStandard", 
      (ins "ArrayRef<Value *>":$operands, "ConversionPatternRewriter&":$rewriter)
    >,
  ];
}


#endif // __PML_ELTWISE_INTERFACES__
