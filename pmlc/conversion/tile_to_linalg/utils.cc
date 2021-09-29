// Copyright 2021, Intel Corporation

#include "mlir/IR/AffineExprVisitor.h"
#include "pmlc/conversion/tile_to_linalg/pass_detail.h"
#include "pmlc/util/extent.h"

namespace pmlc::conversion::tile_to_linalg {

namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT
using namespace pmlc; // NOLINT

TileToLinalgTypeConverter::TileToLinalgTypeConverter() {
  addConversion([](FunctionType type) { return type; });
  addConversion([](FloatType type) { return type; });
  addConversion([](IntegerType type) { return tile::toSignlessType(type); });
  addConversion([](IndexType type) { return type; });
  addConversion([](MemRefType type) { return type; });
  addConversion([this](RankedTensorType type) {
    Type elementType = type.getElementType();
    Type newType = convertType(elementType);
    assert(newType && "could not convert type");
    return RankedTensorType::get(type.getShape(), newType);
  });
}

Value createCastOp(OpBuilder &builder, Location loc, Value from,
                   bool fromSigned, Type intoType, bool intoSigned) {
  Type fromType = from.getType();
  if (fromType == intoType) {
    return from;
  }
  if (auto intoFloatType = intoType.dyn_cast<FloatType>()) {
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (fromFloatType.getWidth() < intoFloatType.getWidth()) {
        // FPExtOp: FloatType -> wider FloatType
        return builder.create<FPExtOp>(loc, from, intoType).getResult();
      }
      // FPTruncOp: FloatType -> narrower FloatType
      return builder.create<FPTruncOp>(loc, from, intoType).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromSigned) {
        // SIToFPOp: IntegerType -> FloatType
        return builder.create<SIToFPOp>(loc, from, intoType).getResult();
      }
      // UIToFPOp: IntegerType -> FloatType
      return builder.create<UIToFPOp>(loc, intoType, from).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType i64Type = builder.getIntegerType(64);
      auto intCastOp = builder.create<IndexCastOp>(loc, from, i64Type);
      return builder.create<SIToFPOp>(loc, intCastOp, intoType).getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (fromSigned) {
          // SignExtendIOp: IntegerType -> wider signed int
          return builder.create<SignExtendIOp>(loc, from, intoType).getResult();
        }
        // ZeroExtendIOp: IntegerType -> wider unsigned int
        return builder.create<ZeroExtendIOp>(loc, from, intoType).getResult();
      }
      // TruncateIOp: IntegerType -> narrower IntegerType
      return builder.create<TruncateIOp>(loc, from, intoType).getResult();
    }
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (intoSigned) {
        // FPToSIOp: FloatType -> signed IntegerType
        return builder.create<FPToSIOp>(loc, from, intoType).getResult();
      }
      // FPToUIOp: FloatType -> unsigned IntegerType
      return builder.create<FPToUIOp>(loc, from, intoType).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType intType = builder.getIntegerType(intoIntType.getWidth());
      return builder.create<IndexCastOp>(loc, from, intType);
    }
  }
  llvm_unreachable("Unsupported cast op");
}

AffineMap updatePaddingMap(AffineMap origMap, const tile::PaddingInfo &padding,
                           MLIRContext *context) {
  assert(padding.lower.size() == origMap.getNumResults());
  SmallVector<AffineExpr, 4> newExprs;
  for (unsigned j = 0; j < origMap.getNumResults(); j++) {
    newExprs.push_back(origMap.getResult(j) + padding.lower[j]);
  }
  return AffineMap::get(origMap.getNumDims(), 0, newExprs, context);
}

class UsedDimsVisitor : public AffineExprVisitor<UsedDimsVisitor> {
public:
  void visitMulExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
  }

  void visitAddExpr(AffineBinaryOpExpr expr) {
    visit(expr.getLHS());
    visit(expr.getRHS());
  }

  void visitDimExpr(AffineDimExpr expr) {
    unsigned pos = expr.getPosition();
    usedDims.insert(pos);
  }

  void visitConstantExpr(AffineConstantExpr expr) {}

  void visitSymbolExpr(AffineSymbolExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: SymbolExpr.");
  }

  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: CeilDivExpr.");
  }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: FloorDivExpr.");
  }

  void visitModExpr(AffineBinaryOpExpr expr) {
    throw std::runtime_error("Unexpected affine expresssion: ModExpr.");
  }

  llvm::SmallSet<int64_t, 4> getUsedDims() { return usedDims; }

private:
  llvm::SmallSet<int64_t, 4> usedDims;
};

llvm::SmallSet<int64_t, 4> getUsedDims(AffineExpr expr) {
  UsedDimsVisitor visitor;
  visitor.visit(expr);
  return visitor.getUsedDims();
}

ContractionMapsAndShapes::ContractionMapsAndShapes(tile::ContractionOp op,
                                                   ValueRange inputs,
                                                   ValueRange outputs,
                                                   ArrayRef<AffineMap> maps)
    : maps(maps), numDims(maps[0].getNumDims()) {
  assert(operands.size() == maps.size());
  MLIRContext *context = op.getContext();
  SmallVector<int64_t> lowerBounds =
      op.lowerBounds().getValue().getConstantResults();
  SmallVector<int64_t> upperBounds =
      op.upperBounds().getValue().getConstantResults();
  for (unsigned i = 0; i < lowerBounds.size(); i++) {
    shape.emplace_back(upperBounds[i] - lowerBounds[i] + 1);
  }
  operands.insert(operands.end(), inputs.begin(), inputs.end());
  operands.insert(operands.end(), outputs.begin(), outputs.end());
}

// This function determines if all the loop dims appear as a single dim in
// shape dims. If not, we need a dummp map to indicate the loop ranges.
bool ContractionMapsAndShapes::needDummyMap() {
  llvm::SmallSet<unsigned, 4> dims;
  for (AffineMap map : maps) {
    assert(numDims == map.getNumDims() &&
           "The input maps have different numbers of dimensions.");
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
        dims.insert(dimExpr.getPosition());
      }
    }
  }
  return dims.size() != numDims;
}

// This function determines if the loop bound inferred by the indexing map
// matches the operand shape. If not, we need to introduce a dynamic dimension
// to bypass the bound check.
bool ContractionMapsAndShapes::needDynamicDim() {
  SmallVector<pmlc::util::Extent, 4> ranges;
  for (auto dim : shape) {
    ranges.emplace_back(pmlc::util::Extent{0, dim - 1});
  }
  for (unsigned i = 0; i < operands.size(); ++i) {
    auto map = maps[i];
    auto opShape = operands[i].getType().cast<RankedTensorType>().getShape();
    auto exprs = maps[i].getResults();
    assert(exprs.size() == opSahpe.size());
    for (unsigned j = 0; j < exprs.size(); ++j) {
      auto extent = pmlc::util::computeExtent(exprs[j], ranges);
      if (extent.max + 1 != opShape[j]) {
        return true;
      }
    }
  }
  return false;
}

linalg::GenericOp createValidGenericOp(
    OpBuilder &builder, Location loc, ContractionMapsAndShapes &info,
    TypeRange resultTypes, ValueRange rawInputs, ValueRange rawOutputs,
    ArrayRef<AffineMap> rawIdxMaps, ArrayRef<StringRef> rawIterTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> body) {
  // GenericOp requires that loop->shape maps can infer shape->loop maps.
  // However the inference function simply calls
  // AffineMap::inversePurmutation(loopToShapeMap) that is too simple to infer
  // shape->loop maps sometimes. So we can't get shape->loop maps and then
  // fail to verify GenericOp sometimes. In this case, we add a redundant
  // tensor and its AffineMap to specify the loop bound explicitly. Then it
  // can pass the GenericOp verification. Later, the redundant tensor could be
  // optimized away because it is useless.
  bool needDummyMap = info.needDummyMap();
  bool needDynamicDim = info.needDynamicDim();
  SmallVector<Value, 4> inputs(rawInputs.begin(), rawInputs.end());
  SmallVector<AffineMap, 4> idxMaps(rawIdxMaps.begin(), rawIdxMaps.end());
  SmallVector<StringRef, 4> iterTypes(rawIterTypes.begin(), rawIterTypes.end());
  SmallVector<NamedAttribute, 1> attr;
  if (needDummyMap || needDynamicDim) {
    SmallVector<int64_t, 4> shape(info.shape);
    SmallVector<Value, 1> dynShape;
    auto context = builder.getContext();
    attr.emplace_back(
        builder.getNamedAttr("dummy_tensor", builder.getUnitAttr()));
    if (needDynamicDim) {
      shape.emplace_back(ShapedType::kDynamicSize);
      auto dynDim =
          builder.create<ConstantIndexOp>(loc, ShapedType::kDynamicSize);
      dynShape.emplace_back(dynDim);
      iterTypes.emplace_back("parallel");
      for (auto &map : idxMaps) {
        map =
            AffineMap::get(map.getNumDims() + 1, 0, map.getResults(), context);
      }
      attr.emplace_back(
          builder.getNamedAttr("skip_bound_check", builder.getUnitAttr()));
    }
    // Create a synthetic tensor with the dimensions of loop bounds.
    auto extraTensor = builder.create<linalg::InitTensorOp>(
        loc, dynShape, shape,
        resultTypes[0].cast<RankedTensorType>().getElementType());
    inputs.insert(inputs.begin(), extraTensor);
    unsigned numDims = needDynamicDim ? info.numDims + 1 : info.numDims;
    AffineMap loopMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    idxMaps.insert(idxMaps.begin(), loopMap);
  }

  // Create the main loop
  return builder.create<linalg::GenericOp>(loc,
                                           /*resultTensorTypes=*/resultTypes,
                                           /*inputs=*/inputs,
                                           /*outputs=*/rawOutputs,
                                           /*indexingMaps=*/idxMaps,
                                           /*iteratorTypes=*/iterTypes,
                                           /*doc=*/"",
                                           /*libraryCall=*/"", body, attr);
}

} // namespace pmlc::conversion::tile_to_linalg
