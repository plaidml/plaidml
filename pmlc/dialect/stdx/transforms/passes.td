#ifndef __PMLC_DIALECT_STDX_PASSES__
#define __PMLC_DIALECT_STDX_PASSES__

include "mlir/Pass/PassBase.td"

def BoundsCheck : FunctionPass<"stdx-check-bounds"> {
  let summary = "Check bounds for Load and Store Ops";
  let constructor = "pmlc::dialect::stdx::createBoundsCheckPass()";
}

#endif // __PMLC_DIALECT_STDX_PASSES__
