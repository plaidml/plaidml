// Copyright 2019, Intel Corporation

#ifndef __PML_ELTWISE_INTERFACES__
#define __PML_ELTWISE_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def EltwiseOpInterface : OpInterface<"EltwiseOp"> {
  let description = "<TODO>";
  let methods = [
    StaticInterfaceMethod<
      "Build a standard op from eltwise on in conversion", 
      "Value*", "buildStandard", 
      (ins "OpBuilder&":$builder, "Location":$loc, "ScalarType":$otype, "ArrayRef<Value *>":$operands)
    >,
  ];
}


#endif // __PML_ELTWISE_INTERFACES__
