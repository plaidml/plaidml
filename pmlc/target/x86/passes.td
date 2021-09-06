#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def CollapseParallelLoops : FunctionPass<"x86-collapse-scf-parallel"> {
  let summary = "Collapse parallel loops to use a single induction variable";
  let constructor = "pmlc::target::x86::createCollapseParallelLoopsPass()";
}

def ConvertClosureToCoro : FunctionPass<"convert-closure-to-coro"> {
  let summary = "Convert closures to coroutines";
  let constructor = "pmlc::target::x86::createLowerClosureToCoroPass()";
  let dependentDialects = ["mlir::async::AsyncDialect"];
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

def PRNGLinking : Pass<"x86-prng-linking", "mlir::ModuleOp"> {
  let summary = "Link prng ops to runtime functions";
  let constructor = "pmlc::target::x86::createPRNGLinkingPass()";
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
