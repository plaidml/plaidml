#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertPXAToAffine : Pass<"x86-convert-pxa-to-affine", "mlir::ModuleOp"> {
  let summary = "Convert PXA dialect to Affine dialect";
  let constructor = "pmlc::target::x86::createLowerPXAToAffinePass()";
}

def ConvertStandardToLLVM : Pass<"x86-convert-std-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert StandardX + XSMM dialects to LLVM dialect";
  let constructor = "pmlc::target::x86::createLowerToLLVMPass()";
}

def TraceLinking : Pass<"x86-trace-linking", "mlir::ModuleOp"> {
  let summary = "Link trace ops to runtime functions";
  let constructor = "pmlc::target::x86::createTraceLinkingPass()";
}

def XSMMStencil : FunctionPass<"x86-affine-stencil-xsmm"> {
  let summary = "Find a tiling for extracting 'micro' GEMMs suitable for XSMM";
  let constructor = "pmlc::target::x86::createXSMMStencilPass()";
}

#endif // __PMLC_TARGET_X86_PASSES__
