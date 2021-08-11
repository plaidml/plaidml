// Copyright 2021, Intel Corporation

#include "mlir/IR/AffineExprVisitor.h"
#include "pmlc/conversion/tile_to_linalg/pass_detail.h"

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
  addConversion([](stdx::ArgpackType type) { return type; });
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

} // namespace pmlc::conversion::tile_to_linalg
