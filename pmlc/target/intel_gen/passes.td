#ifndef __PMLC_TARGET_INTEL_GEN_PASSES__
#define __PMLC_TARGET_INTEL_GEN_PASSES__

include "mlir/Pass/PassBase.td"

def IntelGenLowerAffine : FunctionPass<"intel-gen-lower-affine"> {
  let summary = "Lower affine -> std + scf, and add GPU mappings ";
  let constructor = "pmlc::target::intel_gen::createIntelGenLowerAffinePass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect"
  ];
}

def ConvertStandardToLLVM : Pass<"intel-gen-convert-std-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Standard + StandardX dialects to LLVM dialect";
  let constructor = "pmlc::target::intel_gen::createConvertStandardToLLVM()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertParallelLoopToGpu : Pass<"intel-gen-convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped scf.parallel ops to gpu launch operations";
  let constructor = "pmlc::target::createParallelLoopToGpuPass()";
  let dependentDialects = ["mlir::AffineDialect", "mlir::gpu::GPUDialect"];
}


#endif

