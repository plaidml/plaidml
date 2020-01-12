// Copyright 2019, Intel Corporation

#ifndef __PML_ELTWISE_INTERFACES__
#define __PML_ELTWISE_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def EltwiseOpInterface : OpInterface<"EltwiseOp">;

#endif // __PML_ELTWISE_INTERFACES__
