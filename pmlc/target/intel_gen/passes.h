#pragma once

#include <memory>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::intel_gen {

std::unique_ptr<mlir::Pass> createIntelGenLowerAffinePass();

std::unique_ptr<mlir::Pass> createAffineIndexPackPass();

std::unique_ptr<mlir::Pass> createConvertStandardToLLVM();

std::unique_ptr<mlir::Pass> createParallelLoopToGpuPass();

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

void pipelineBuilder(mlir::OpPassManager &pm);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/intel_gen/passes.h.inc"

} // namespace pmlc::target::intel_gen
