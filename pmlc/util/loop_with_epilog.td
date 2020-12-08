// Copyright 2020, Intel Corporation

#ifndef __PML_LOOP_WITH_EPILOG__
#define __PML_LOOP_WITH_EPILOG__

include "mlir/IR/OpBase.td"

def LoopWithEpilogInterface : OpInterface<"LoopWithEpilogInterface"> {
  let description = [{
    Extends LoopLikeOpInterface to support loops with support for moving
    operations to an epilog region.
  }];
  let cppNamespace = "::pmlc::util";

  let methods = [
    InterfaceMethod<[{
        Moves the given vector of operations to the loop epilog.
        The vector is sorted topologically.
      }],
      "LogicalResult", "moveToLoopEpilog", (ins "ArrayRef<Operation *>":$ops)
    >,
  ];
}

#endif // __PML_LOOP_WITH_EPILOG__
