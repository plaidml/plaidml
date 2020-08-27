#pragma once

#include <memory>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::intel_gen {

std::unique_ptr<mlir::Pass> createIntelGenLowerAffinePass();

std::unique_ptr<mlir::Pass> createParallelLoopToGpuPass();

void pipelineBuilder(mlir::OpPassManager &pm);

} // namespace pmlc::target::intel_gen
