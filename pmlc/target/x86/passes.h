#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class LLVMTypeConverter;
class LowerToLLVMOptions;
class MLIRContext;
class OpPassManager;
class RewritePatternSet;
} // namespace mlir

namespace pmlc::target::x86 {

std::unique_ptr<mlir::Pass> createCollapseParallelLoopsPass();

std::unique_ptr<mlir::Pass> createFoldConstantCastPass();

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createProfileKernelsPass();

std::unique_ptr<mlir::Pass> createProfileLinkingPass();

std::unique_ptr<mlir::Pass> createPRNGLinkingPass();

std::unique_ptr<mlir::Pass> createShapeAnalysisPass();

std::unique_ptr<mlir::Pass> createReorderLayoutsPass();

std::unique_ptr<mlir::Pass> createReorderWeightLayoutsPass();

std::unique_ptr<mlir::Pass> createStencilTppGemmPass();

std::unique_ptr<mlir::Pass> createStencilTppUnaryPass();

std::unique_ptr<mlir::Pass> createStencilTppBinaryPass();

std::unique_ptr<mlir::Pass> createStencilSplitPass();

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

void populatePXAGemmToXSMMConversionPatterns(mlir::RewritePatternSet &patterns);

void populateXSMMToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
