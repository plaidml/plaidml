// Copyright 2019, Intel Corporation

#ifndef __PML_TILE_INTERFACES__
#define __PML_TILE_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def SpecialOpInterface : OpInterface<"SpecialOp">;

#endif // __PML_TILE_INTERFACES__
