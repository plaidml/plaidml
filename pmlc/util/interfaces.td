// Copyright 2020 Intel Corporation

#ifndef __PML_UTIL_INTERFACES__
#define __PML_UTIL_INTERFACES__

include "mlir/IR/OpBase.td"

def MaterializeOperandsOpInterface : OpInterface<"MaterializeOperandsOpInterface"> {
  let description = [{
    TODO
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"",
      /*retTy=*/"LogicalResult",
      /*methodName=*/"materializeOperands",
      /*args=*/(ins "OpBuilder&":$builder),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return failure(); }]
    >
  ];
}

#endif  // __PML_UTIL_INTERFACES__
