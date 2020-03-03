// Copyright 2020 Intel Corporation

#pragma once

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;
} // namespace mlir

namespace pmlc::target::x86 {

void populateXSMMConversionPatterns(mlir::OwningRewritePatternList &patterns,
                                    mlir::MLIRContext *ctx);

} // namespace pmlc::target::x86
