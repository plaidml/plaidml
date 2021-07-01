#pragma once

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::target::x86 {

void pipelineBuilder(mlir::OpPassManager &pm);

void registerTarget();

} // namespace pmlc::target::x86
