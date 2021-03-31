// Copyright 2020, Intel Corporation
#ifndef __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__
#define __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertCompToLLVM : Pass<"pmlc-convert-comp-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert comp operations to runtime calls";
  let constructor = "pmlc::conversion::comp_to_llvm::createConvertCompToLLVMPass()";
  let options = [
    Option<"prefix", "comp-llvm-prefix", "std::string",
           /*default=*/"\"\"", "String prefix to use for lowered comp funtions">,
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::StandardOpsDialect"
  ];
}

#endif // __PMLC_CONVERSION_COMP_TO_LLVM_PASSES__
