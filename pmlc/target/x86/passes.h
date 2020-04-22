#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::target::x86 {

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

std::unique_ptr<mlir::Pass> createXSMMStencilPass();

} // namespace pmlc::target::x86
