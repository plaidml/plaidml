#pragma once

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::intel_gen {

void pipelineBuilder(mlir::OpPassManager &pm);

} // namespace pmlc::target::intel_gen
