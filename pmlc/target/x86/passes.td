#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def CollapseParallelLoops : Pass<"x86-collapse-scf-parallel", "mlir::func::FuncOp"> {
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

def FoldConstantCast : Pass<"x86-fold-constant-cast", "mlir::func::FuncOp"> {
  let summary = "Materialize casts of constants to target type";
  let constructor = "pmlc::target::x86::createFoldConstantCastPass()";
}

def ProfileKernels : Pass<"x86-profile-kernels", "mlir::ModuleOp"> {
  let summary = "Inject instrumentation points before/after kernel ops";
  let constructor = "pmlc::target::x86::createProfileKernelsPass()";
}

def ProfileLinking : Pass<"x86-profile-linking", "mlir::ModuleOp"> {
  let summary = "Link profile ops to runtime functions";
  let constructor = "pmlc::target::x86::createProfileLinkingPass()";
}

def PRNGLinking : Pass<"x86-prng-linking", "mlir::ModuleOp"> {
  let summary = "Link prng ops to runtime functions";
  let constructor = "pmlc::target::x86::createPRNGLinkingPass()";
}

def ReorderLayouts : Pass<"x86-reorder-layouts", "mlir::func::FuncOp"> {
  let summary = "Optimize data layouts for efficient reading";
  let constructor = "pmlc::target::x86::createReorderLayoutsPass()";
  let options = [
    Option<"blockSize", "block-size", "int64_t", /*default=*/"32",
           "Data block size">
  ];
  let dependentDialects = [
    "dialect::linalgx::LinalgXDialect",
  ];
}

def ReorderWeightLayouts : Pass<"x86-reorder-weight-layouts", "mlir::func::FuncOp"> {
  let summary = "Optimize weight layouts for efficient reading";
  let constructor = "pmlc::target::x86::createReorderWeightLayoutsPass()";
  let options = [
    Option<"blockSize", "block-size", "int64_t", /*default=*/"32",
           "Data block size">
  ];
  let dependentDialects = [
    "dialect::linalgx::LinalgXDialect",
  ];
}

def StencilTppGemm : Pass<"x86-stencil-tpp-gemm", "mlir::func::FuncOp"> {
  let summary = "Find a stencil for TPP GEMM ops";
  let constructor = "pmlc::target::x86::createStencilTppGemmPass()";
  let options = [
    Option<"numThreads", "threads", "unsigned", /*default=*/"0",
           "Number of threads">,
    Option<"isBatched", "batched", "bool", /*default=*/"false",
           "Enable strided batching over k dimension of GEMM">
  ];
}

def StencilTppUnary : Pass<"x86-stencil-tpp-unary", "mlir::func::FuncOp"> {
  let summary = "Find a stencil for TPP unary ops";
  let constructor = "pmlc::target::x86::createStencilTppUnaryPass()";
}

def StencilTppBinary : Pass<"x86-stencil-tpp-binary", "mlir::func::FuncOp"> {
  let summary = "Find a stencil for TPP binary ops";
  let constructor = "pmlc::target::x86::createStencilTppBinaryPass()";
}

def StencilSplit : Pass<"x86-stencil-tpp-split", "mlir::func::FuncOp"> {
  let summary = "Split back to back unary/binary stencils";
  let constructor = "pmlc::target::x86::createStencilSplitPass()";
}

def TppCombine : Pass<"x86-tpp-combine", "mlir::func::FuncOp"> {
  let summary = "Combine identity->gemm->relu pattern";
  let constructor = "pmlc::target::x86::createTppCombinePass()";
}

def TraceLinking : Pass<"x86-trace-linking", "mlir::ModuleOp"> {
  let summary = "Link trace ops to runtime functions";
  let constructor = "pmlc::target::x86::createTraceLinkingPass()";
}

#endif // __PMLC_TARGET_X86_PASSES__
