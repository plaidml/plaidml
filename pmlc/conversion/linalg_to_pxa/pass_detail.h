// Copyright 2021, Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"

namespace pmlc::conversion::linalg_to_pxa {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/linalg_to_pxa/passes.h.inc"

struct LinalgToPXATypeConverter : public mlir::TypeConverter {
  LinalgToPXATypeConverter();
};

using GenericOpBodyBuilder =
    llvm::function_ref<void(mlir::OpBuilder &, unsigned, mlir::ValueRange)>;

mlir::linalg::GenericOp createGenericOp(
    mlir::OpBuilder &builder, mlir::Operation *op, mlir::TypeRange outputTypes,
    mlir::ValueRange inputs, mlir::ValueRange outputs, unsigned numIdxs,
    mlir::ArrayRef<mlir::AffineMap> maps, GenericOpBodyBuilder bodyBuilder);

void populateLinalgToPXASpecialPatterns(mlir::RewritePatternSet &patterns);

void populateLinalgTensorCollapseOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns);

void populateLinalgTensorExpandOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns);

void populateLinalgPoolingOpGeneralizationPatterns(
    mlir::RewritePatternSet &patterns);

} // namespace pmlc::conversion::linalg_to_pxa
