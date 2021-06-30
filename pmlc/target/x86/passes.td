#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def ConvertPXAToAffine : Pass<"x86-convert-pxa-to-affine", "mlir::ModuleOp"> {
  let summary = "Convert PXA dialect to Affine dialect";
  let constructor = "pmlc::target::x86::createLowerPXAToAffinePass()";
  let dependentDialects = [
    "dialect::xsmm::XSMMDialect",
    "mlir::AffineDialect",
  ];
}

def ConvertStandardToLLVM : Pass<"x86-convert-std-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert StandardX + XSMM dialects to LLVM dialect";
  let constructor = "pmlc::target::x86::createLowerToLLVMPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def PRNGLinking : Pass<"x86-prng-linking", "mlir::ModuleOp"> {
  let summary = "Link prng ops to runtime functions";
  let constructor = "pmlc::target::x86::createPRNGLinkingPass()";
}

def StencilTppUnary : FunctionPass<"x86-stencil-tpp-unary"> {
  let summary = "Find a stencil for TPP unary ops";
  let constructor = "pmlc::target::x86::createStencilTppUnaryPass()";
}

def TraceLinking : Pass<"x86-trace-linking", "mlir::ModuleOp"> {
  let summary = "Link trace ops to runtime functions";
  let constructor = "pmlc::target::x86::createTraceLinkingPass()";
}

def XSMMStencil : FunctionPass<"x86-affine-stencil-xsmm"> {
  let summary = "Find a tiling for extracting 'micro' GEMMs suitable for XSMM";
  let constructor = "pmlc::target::x86::createXSMMStencilPass()";
  let options = [
    Option<"numThreads", "threads", "unsigned", /*default=*/"0",
           "Number of threads">,
    Option<"isBatched", "batched", "bool", /*default=*/"false",
           "Enable strided batching over k dimension of GEMM">
  ];
}

#endif // __PMLC_TARGET_X86_PASSES__
