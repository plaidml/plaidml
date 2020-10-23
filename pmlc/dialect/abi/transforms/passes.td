#ifndef __PMLC_DIALECT_ABI_PASSES__
#define __PMLC_DIALECT_ABI_PASSES__

include "mlir/Pass/PassBase.td"

def LowerToABIPass : Pass<"lower-to-abi", "mlir::ModuleOp"> {
  let summary = "Packages the network as a loop for repeated uses";
  let constructor = "pmlc::dialect::abi::createLowerToABIPass()";
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

def DenormalizeConstantsPass : Pass<"abi-denormalize-constants", "pmlc::dialect::abi::LoopOp"> {
  let summary = "Denormalizes constants from loop initialization";
  let constructor = "pmlc::dialect::abi::createDenormalizeConstantsPass()";
  let dependentDialects = [
    "pmlc::dialect::abi::ABIDialect"
  ];
}

#endif // __PMLC_DIALECT_ABI_PASSES__
