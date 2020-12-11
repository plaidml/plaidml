#ifndef __PMLC_TRANSFORMS__
#define __PMLC_TRANSFORMS__

include "mlir/Pass/PassBase.td"

def HoistingPass : Pass<"pmlc-hoisting", "mlir::ModuleOp"> {
  let summary = "Hoists operations out of loops";
  let description = [{
    Similar to the standard MLIR loop-invariant code motion, this pass hoists
    operations out of loop operations.  A key difference is that when the loop
    supports hoisting finalization operations (after the loop body), this pass
    also considers operations with memory side-effects: it will pair up
    operations that allocate and free a resource, and move them into the loop
    initialization and finalization regions as a pair.
  }];
  let constructor = "pmlc::transforms::createHoistingPass()";
}

#endif // __PMLC_TRANSFORMS__
