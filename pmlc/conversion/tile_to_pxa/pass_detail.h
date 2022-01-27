#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/layer/ir/ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"

namespace pmlc::conversion::tile_to_pxa {

struct TileToPXATypeConverter : public mlir::TypeConverter {
  TileToPXATypeConverter();
};

void populateTileToPXASpecialPatterns(mlir::RewritePatternSet &patterns);

mlir::Value createCastOp(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value from, bool fromSigned, mlir::Type intoType,
                         bool intoSigned);

mlir::Value
buildSimpleStore(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value scalar, mlir::Value memRef,
                 mlir::Optional<pmlc::dialect::tile::PaddingInfo> maybePadding);

void updateAffineMap(mlir::Operation *in,
                     const pmlc::dialect::tile::PaddingInfo &padding);

llvm::APFloat convertFloatUsingType(llvm::APFloat value, mlir::FloatType type);

mlir::Value createInit(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Type type, pmlc::util::AggregationKind agg);

mlir::Value buildBroadcastLoad(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value operand,
    unsigned outRank,
    mlir::Optional<pmlc::dialect::tile::PaddingInfo> maybePadding = llvm::None);

struct BufferAllocator {
  mlir::Value resultMemRef;
  mlir::RankedTensorType rankedTensorType;
  mlir::MemRefType memRefType;
  mlir::Type elementType;

  BufferAllocator(mlir::OpBuilder &builder, mlir::Operation *op,
                  mlir::Type resultType);
};

} // namespace pmlc::conversion::tile_to_pxa
