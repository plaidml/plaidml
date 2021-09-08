// Copyright 2021, Intel Corporation

#pragma once

#include "llvm/ADT/SmallSet.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"

namespace pmlc::conversion::tile_to_linalg {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/tile_to_linalg/passes.h.inc"

struct TileToLinalgTypeConverter : public mlir::TypeConverter {
  TileToLinalgTypeConverter();
};

mlir::Value createCastOp(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value from, bool fromSigned, mlir::Type intoType,
                         bool intoSigned);

mlir::Value
buildSimpleStore(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value scalar, mlir::Value memRef,
                 mlir::Optional<pmlc::dialect::tile::PaddingInfo> maybePadding);

mlir::AffineMap
updatePaddingMap(mlir::AffineMap origMap,
                 const pmlc::dialect::tile::PaddingInfo &padding,
                 mlir::MLIRContext *context);

llvm::SmallSet<int64_t, 4> getUsedDims(mlir::AffineExpr expr);

void populateTileToLinalgSpecialPatterns(mlir::RewritePatternSet &patterns);

} // namespace pmlc::conversion::tile_to_linalg
