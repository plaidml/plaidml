#pragma once

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class LowerToLLVMOptions;
class OwningRewritePatternList;
class Pass;
} // namespace mlir

namespace pmlc::target::x86 {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

std::unique_ptr<mlir::Pass> createXSMMStencilPass();

void populateXSMMToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns,
    const mlir::LowerToLLVMOptions &options);

void registerPassPipeline();

} // namespace pmlc::target::x86
