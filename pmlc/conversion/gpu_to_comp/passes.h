// Copyright 2020, Intel Corporation
#pragma once

#include <memory>

namespace pmlc::dialect::comp {
class ExecEnvType;
} // namespace pmlc::dialect::comp

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu_to_comp {

void populateGpuToCompPatterns(
    mlir::MLIRContext *context,
    const pmlc::dialect::comp::ExecEnvType &execEnvType,
    mlir::OwningRewritePatternList &patterns);

std::unique_ptr<mlir::Pass> createConvertGpuToCompPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/gpu_to_comp/passes.h.inc"

} // namespace pmlc::conversion::gpu_to_comp
