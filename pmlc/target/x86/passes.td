#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def CollapseParallelLoops : FunctionPass<"x86-collapse-scf-parallel"> {
  let summary = "Collapse parallel loops to use a single induction variable";
  let constructor = "pmlc::target::x86::createCollapseParallelLoopsPass()";
}

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

def FoldConstantCast : FunctionPass<"fold-constant-cast"> {
  let summary = "Materialize casts of constants to target type";
  let constructor = "pmlc::target::x86::createFoldConstantCastPass()";
}

def PRNGLinking : Pass<"x86-prng-linking", "mlir::ModuleOp"> {
  let summary = "Link prng ops to runtime functions";
  let constructor = "pmlc::target::x86::createPRNGLinkingPass()";
}

def ReorderLayouts : FunctionPass<"x86-reorder-layouts"> {
  let summary = "Optimize data layouts for efficient reading";
  let constructor = "pmlc::target::x86::createReorderLayoutsPass()";
  let options = [
    Option<"blockSize", "block-size", "int64_t", /*default=*/"16",
           "Data block size">
  ];
}

def StencilTppGemm : FunctionPass<"x86-stencil-tpp-gemm"> {
  let summary = "Find a stencil for TPP GEMM ops";
  let constructor = "pmlc::target::x86::createStencilTppGemmPass()";
  let options = [
    Option<"numThreads", "threads", "unsigned", /*default=*/"0",
           "Number of threads">,
    Option<"isBatched", "batched", "bool", /*default=*/"false",
           "Enable strided batching over k dimension of GEMM">
  ];
}

def StencilTppUnary : FunctionPass<"x86-stencil-tpp-unary"> {
  let summary = "Find a stencil for TPP unary ops";
  let constructor = "pmlc::target::x86::createStencilTppUnaryPass()";
}

def TraceLinking : Pass<"x86-trace-linking", "mlir::ModuleOp"> {
  let summary = "Link trace ops to runtime functions";
  let constructor = "pmlc::target::x86::createTraceLinkingPass()";
}

#endif // __PMLC_TARGET_X86_PASSES__
