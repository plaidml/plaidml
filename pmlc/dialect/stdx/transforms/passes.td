#ifndef __PMLC_DIALECT_STDX_PASSES__
#define __PMLC_DIALECT_STDX_PASSES__

include "mlir/Pass/PassBase.td"

def BoundsCheck : FunctionPass<"stdx-check-bounds"> {
  let summary = "Check bounds for Load and Store Ops";
  let constructor = "pmlc::dialect::stdx::createBoundsCheckPass()";
}

def I1StorageToI32 : FunctionPass<"stdx-i1-storage-to-i32"> {
  let summary = "Convert i1 storage type to i32";
  let constructor = "pmlc::dialect::stdx::createI1StorageToI32Pass()";
}

#endif // __PMLC_DIALECT_STDX_PASSES__
