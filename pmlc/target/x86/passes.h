#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class LLVMTypeConverter;
class LowerToLLVMOptions;
class MLIRContext;
class OpPassManager;
class OwningRewritePatternList;
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

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
