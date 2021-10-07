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

struct OpMapsAndShapes {
  // The constructor for ContractionOp. We may add constructors for other
  // operations.
  OpMapsAndShapes(pmlc::dialect::tile::ContractionOp op,
                  mlir::ValueRange inputs, mlir::ValueRange outputs,
                  llvm::ArrayRef<mlir::AffineMap> maps);

  // This function determines if all the loop dims appear as a single dim in
  // shape dims. If not, we need a dummp map to indicate the loop ranges.
  bool needDummyMap();

  // This function determines if the loop bound inferred by the indexing map
  // matches the operand shape. If not, we need to introduce a dynamic dimension
  // to bypass the bound check.
  bool needDynamicDim();

  llvm::ArrayRef<mlir::AffineMap> maps;
  llvm::SmallVector<mlir::Value, 4> operands;
  llvm::SmallVector<int64_t, 4> shape;
  unsigned numDims;
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

mlir::linalg::GenericOp
createValidGenericOp(mlir::OpBuilder &builder, mlir::Location loc,
                     OpMapsAndShapes &info, mlir::TypeRange resultTypes,
                     mlir::ValueRange rawInputs, mlir::ValueRange rawOutputs,
                     llvm::ArrayRef<mlir::AffineMap> rawIdxMaps,
                     llvm::ArrayRef<llvm::StringRef> rawIterTypes,
                     llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                             mlir::ValueRange)>
                         body);

} // namespace pmlc::conversion::tile_to_linalg
