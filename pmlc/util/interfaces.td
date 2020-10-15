// Copyright 2020 Intel Corporation

#ifndef __PML_UTIL_INTERFACES__
#define __PML_UTIL_INTERFACES__

include "mlir/IR/OpBase.td"

def MaterializeOperandsOpInterface : OpInterface<"MaterializeOperandsOpInterface"> {
  let description = [{
    TODO
  }];

  let cppNamespace = "pmlc::util";

  let methods = [
    InterfaceMethod<
      /*desc=*/"",
      /*retTy=*/"mlir::LogicalResult",
      /*methodName=*/"materializeOperands",
      /*args=*/(ins "mlir::OpBuilder&":$builder),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return mlir::failure(); }]
    >
  ];
}

#endif  // __PML_UTIL_INTERFACES__
