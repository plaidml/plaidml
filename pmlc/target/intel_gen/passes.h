#pragma once

#include <memory>

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::intel_gen {

std::unique_ptr<mlir::Pass> createIntelGenLowerAffinePass();

void pipelineBuilder(mlir::OpPassManager &pm);

} // namespace pmlc::target::intel_gen
