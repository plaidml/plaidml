#ifndef __PMLC_DIALECT_STDX_PASSES__
#define __PMLC_DIALECT_STDX_PASSES__

include "mlir/Pass/PassBase.td"

def BoundsCheck : Pass<"stdx-check-bounds", "mlir::func::FuncOp"> {
  let summary = "Check bounds for Load and Store Ops";
  let constructor = "pmlc::dialect::stdx::createBoundsCheckPass()";
}

def MainClosure : Pass<"stdx-main-closure", "mlir::ModuleOp"> {
  let summary = "Wrap a main function in a closure";
  let constructor = "pmlc::dialect::stdx::createMainClosurePass()";
  let dependentDialects = ["pmlc::dialect::stdx::StdXDialect"];
}

def SplitClosure : Pass<"stdx-split-closure", "mlir::ModuleOp"> {
  let summary = "Split a closure into init/main/fini functions";
  let constructor = "pmlc::dialect::stdx::createSplitClosurePass()";
}

#endif // __PMLC_DIALECT_STDX_PASSES__
