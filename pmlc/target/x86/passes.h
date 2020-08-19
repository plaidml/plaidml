#pragma once

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class LowerToLLVMOptions;
class MLIRContext;
class OpPassManager;
class OwningRewritePatternList;
class Pass;
} // namespace mlir

namespace pmlc::target::x86 {

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

std::unique_ptr<mlir::Pass> createXSMMStencilPass();

void populatePXAGemmToXSMMConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx);

void populatePXAPrngToAffineConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *ctx);

void populateXSMMToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns);

void pipelineBuilder(mlir::OpPassManager &pm);

} // namespace pmlc::target::x86
