#ifndef __PMLC_DIALECT_ABI_PASSES__
#define __PMLC_DIALECT_ABI_PASSES__

include "mlir/Pass/PassBase.td"

def AddABILoopPass : Pass<"abi-add-loop", "mlir::ModuleOp"> {
  let summary = "Adds an abstract loop around the network, matching the way the network will be run";
  let constructor = "pmlc::dialect::abi::createAddABILoopPass()";
  let options = [
    Option<"networkMain", "main", "std::string",
           /*default=*/"::pmlc::util::kEntrypoint",
           "The name of the network's main entrypoint">,
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "pmlc::dialect::abi::ABIDialect"
  ];
}

def LowerToABIPass : Pass<"abi-lower", "mlir::ModuleOp"> {
  let summary = "Creates the standard ABI entrypoints";
  let constructor = "pmlc::dialect::abi::createLowerToABIPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "pmlc::dialect::abi::ABIDialect"
  ];
}

#endif // __PMLC_DIALECT_ABI_PASSES__
