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

mlir::AffineMap
updatePaddingMap(mlir::AffineMap origMap,
                 const pmlc::dialect::tile::PaddingInfo &padding,
                 mlir::MLIRContext *context);

mlir::tensor::ExtractSliceOp
sliceTensor(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source,
            const pmlc::dialect::tile::PaddingInfo &padding);

llvm::SmallSet<int64_t, 4> getUsedDims(mlir::AffineExpr expr);

} // namespace pmlc::conversion::tile_to_linalg
