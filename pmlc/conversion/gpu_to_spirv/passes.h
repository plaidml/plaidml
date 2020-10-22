// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
class MLIRContext;
class OwningRewritePatternList;
class SPIRVTypeConverter;
} // namespace mlir

namespace pmlc::conversion::gpu_to_spirv {

void populateStdxToSPIRVPatterns(mlir::MLIRContext *context,
                                 mlir::SPIRVTypeConverter &typeConverter,
                                 mlir::OwningRewritePatternList &patterns);

void populateStdxToSPIRVGLSLPatterns(mlir::MLIRContext *context,
                                     mlir::SPIRVTypeConverter &typeConverter,
                                     mlir::OwningRewritePatternList &patterns);

/// Adds patterns that perform conversion from GLSL spirv operations
/// to chain of core spirv operations.
void populateCustomGLSLToStdSpirvPatterns(
    mlir::MLIRContext *context, mlir::SPIRVTypeConverter &typeConverter,
    mlir::OwningRewritePatternList &patterns);

void populateCustomStdToOCLSpirvPatterns(
    mlir::MLIRContext *context, mlir::SPIRVTypeConverter &typeConverter,
    mlir::OwningRewritePatternList &patterns);

std::unique_ptr<mlir::Pass> createGPUToSPIRVCustomPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/gpu_to_spirv/passes.h.inc"

} // namespace pmlc::conversion::gpu_to_spirv
