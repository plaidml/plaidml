#ifndef __PMLC_TARGET_X86_PASSES__
#define __PMLC_TARGET_X86_PASSES__

include "mlir/Pass/PassBase.td"

def TraceLinking : Pass<"x86-trace-linking", "mlir::ModuleOp"> {
  let summary = "Link trace ops to runtime functions";
  let constructor = "pmlc::target::x86::createTraceLinkingPass()";
}

def XSMMLowering : FunctionPass<"x86-xsmm"> {
  let summary = "XSMM to standard conversion";
  let constructor = "pmlc::target::x86::createXSMMLoweringPass()";
}

def XSMMStencil : FunctionPass<"x86-affine-stencil-xsmm"> {
  let summary = "Find a tiling for extracting 'micro' GEMMs suitable for XSMM";
  let constructor = "pmlc::target::x86::createXSMMStencilPass()";
}

#endif // __PMLC_TARGET_X86_PASSES__
