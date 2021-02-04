// Copyright 2020 Intel Corporation
#include "pmlc/util/layout.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/tile/ir/util.h"

#include "mlir/IR/BuiltinTypes.h"

namespace pmlc {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

MLFramework getMLFramework(StringRef opName) {
  StringRef nGraphPrefix = "ng.";
  if (opName.substr(0, nGraphPrefix.size()) == nGraphPrefix) {
    return MLFramework::NGraph;
  } else {
    return MLFramework::Default;
  }
}

TensorLayout getLayoutType(MLFramework framework, StringRef opName,
                           bool isConst) {
  // TODO: Create a common list of primitives that would be used here and in the
  // plugins so we know they are consistent. For now use namings from OV plugin
  // Const are used to identify the primitives weights or parameters, rest is
  // considered as main flow data type
  if (framework == MLFramework::NGraph) {
    if ((opName.find("Convolution") != StringRef::npos) && isConst) {
      return TensorLayout::KCX;
    } else {
      return TensorLayout::NCX;
    }
  }

  return TensorLayout::NXC;
}

MemRefType updateMemRefWithLayoutMap(MLIRContext *context,
                                     RankedTensorType memrefType,
                                     Type elementType, TensorLayout layout) {
  auto rankedTensorType = pmlc::dialect::tile::getRankedTensorType(memrefType);
  auto originalShape = rankedTensorType.getShape();
  auto shape = llvm::to_vector<8>(originalShape);

  auto outRank = rankedTensorType.getRank();
  auto spatialNum = outRank == 4 ? 2 : 0;

  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(outRank);
  if (layout == TensorLayout::NCX) {
    dimExprs.push_back(mlir::getAffineDimExpr(0, context));
    for (auto spat = 0; spat < spatialNum; spat++) {
      dimExprs.push_back(
          mlir::getAffineDimExpr(outRank - spatialNum + spat, context));
    }
    dimExprs.push_back(mlir::getAffineDimExpr(1, context));
  } else if (layout == TensorLayout::KCX) {
    for (auto spat = 0; spat < spatialNum; spat++) {
      dimExprs.push_back(
          mlir::getAffineDimExpr(outRank - spatialNum + spat, context));
    }
    dimExprs.push_back(mlir::getAffineDimExpr(1, context));
    dimExprs.push_back(mlir::getAffineDimExpr(0, context));
  }
  auto idMap = AffineMap::get(outRank, 0, dimExprs, context);
  return MemRefType::get(shape, elementType, {idMap});
}

} // namespace pmlc
