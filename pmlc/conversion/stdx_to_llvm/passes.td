#ifndef __PMLC_CONVERSION_STDX_TO_LLVM_PASSES__
#define __PMLC_CONVERSION_STDX_TO_LLVM_PASSES__

include "mlir/Pass/PassBase.td"

def LowerToLLVM : Pass<"convert-stdx-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert stdx to the LLVM dialect";
  let constructor = "pmlc::conversion::stdx_to_llvm::createLowerToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // __PMLC_CONVERSION_STDX_TO_LLVM_PASSES__
