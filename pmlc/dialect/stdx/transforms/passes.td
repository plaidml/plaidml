#ifndef __PMLC_DIALECT_STDX_PASSES__
#define __PMLC_DIALECT_STDX_PASSES__

include "mlir/Pass/PassBase.td"

def BoundsCheck : FunctionPass<"stdx-check-bounds"> {
  let summary = "Check bounds for Load and Store Ops";
  let constructor = "pmlc::dialect::stdx::createBoundsCheckPass()";
}

def SplitMain : Pass<"stdx-split-main", "mlir::ModuleOp"> {
  let summary = "Split main into init/main/fini blocks";
  let constructor = "pmlc::dialect::stdx::createSplitMainPass()";
  let dependentDialects = ["pmlc::dialect::stdx::StdXDialect"];
}

#endif // __PMLC_DIALECT_STDX_PASSES__
