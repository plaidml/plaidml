#ifndef __PMLC_TRANSFORM_PASSES__
#define __PMLC_TRANSFORM_PASSES__

include "mlir/Pass/PassBase.td"

def BoxingPass : Pass<"llvm-boxing", "mlir::ModuleOp"> {
  let summary = "Creates wrappers that box function inputs and outputs";
  let constructor = "pmlc::transforms::createBoxingPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

def MakeEntrypointsPass : Pass<"pmlc-make-entrypoints", "mlir::ModuleOp"> {
  let summary = "Creates the standard PlaidML entrypoints";
  let constructor = "pmlc::transforms::createMakeEntrypointsPass()";
  let options = [
    Option<"networkMain", "main", "std::string",
           /*default=*/"::pmlc::util::kEntrypoint",
           "The name of the network's main entrypoint">,
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // __PMLC_TRANSFORM_PASSES__
