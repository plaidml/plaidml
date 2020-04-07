// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
class Pass;
} // namespace mlir

namespace pmlc::target::x86 {

void populateXSMMConversionPatterns(mlir::OwningRewritePatternList &patterns,
                                    mlir::MLIRContext *ctx);

std::unique_ptr<mlir::Pass> createXSMMLoweringPass();

} // namespace pmlc::target::x86
