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

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createOpenMPWorkaroundPass();

std::unique_ptr<mlir::Pass> createPRNGLinkingPass();

std::unique_ptr<mlir::Pass> createTppPatternsPass();

std::unique_ptr<mlir::Pass> createTraceLinkingPass();

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

std::unique_ptr<mlir::Pass> createXSMMStencilPass();

void populatePXAGemmToXSMMConversionPatterns(mlir::RewritePatternSet &patterns);

void populateXSMMToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                          mlir::RewritePatternSet &patterns);

void pipelineBuilder(mlir::OpPassManager &pm);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
