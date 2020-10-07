#ifndef __PMLC_TRANSFORM_PASSES__
#define __PMLC_TRANSFORM_PASSES__

include "mlir/Pass/PassBase.td"

def MakeEntrypointsPass : Pass<"pmlc-make-entrypoints", "mlir::ModuleOp"> {
  let summary = "Creates the standard PlaidML entrypoints";
  let constructor = "pmlc::transforms::createMakeEntrypointsPass()";
  let options = [
    Option<"networkMain", "main", "std::string",
           /*default=*/"::pmlc::kEntrypoint",
           "The name of the network's main entrypoint">,
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif // __PMLC_TRANSFORM_PASSES__
