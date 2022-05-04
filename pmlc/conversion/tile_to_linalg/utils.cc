// Copyright 2021, Intel Corporation

#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "pmlc/conversion/tile_to_linalg/pass_detail.h"
#include "pmlc/util/extent.h"

namespace pmlc::conversion::tile_to_linalg {

namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT
using namespace pmlc; // NOLINT

// This is needed to workaround multiple definition linker issues on debug
// builds in the CI environment.
static constexpr int64_t kDynamicSize = -1;

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
        // arith::ExtFOp: FloatType -> wider FloatType
        return builder.create<arith::ExtFOp>(loc, from, intoType).getResult();
      }
      // arith::TruncFOp: FloatType -> narrower FloatType
      return builder.create<arith::TruncFOp>(loc, from, intoType).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromSigned) {
        // arith::SIToFPOp: IntegerType -> FloatType
        return builder.create<arith::SIToFPOp>(loc, from, intoType).getResult();
      }
      // arith::UIToFPOp: IntegerType -> FloatType
      return builder.create<arith::UIToFPOp>(loc, intoType, from).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType i64Type = builder.getIntegerType(64);
      auto intCastOp = builder.create<arith::IndexCastOp>(loc, from, i64Type);
      return builder.create<arith::SIToFPOp>(loc, intCastOp, intoType).getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (fromSigned) {
          // arith::ExtSIOp: IntegerType -> wider signed int
          return builder.create<arith::ExtSIOp>(loc, from, intoType).getResult();
        }
        // arith::ExtUIOp: IntegerType -> wider unsigned int
        return builder.create<arith::ExtUIOp>(loc, from, intoType).getResult();
      }
      // arith::TruncIOp: IntegerType -> narrower IntegerType
      return builder.create<arith::TruncIOp>(loc, from, intoType).getResult();
    }
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (intoSigned) {
        // arith::FPToSIOp: FloatType -> signed IntegerType
        return builder.create<arith::FPToSIOp>(loc, from, intoType).getResult();
      }
      // arith::FPToUIOp: FloatType -> unsigned IntegerType
      return builder.create<arith::FPToUIOp>(loc, from, intoType).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType intType = builder.getIntegerType(intoIntType.getWidth());
      return builder.create<arith::IndexCastOp>(loc, from, intType);
    }
  }
  llvm_unreachable("Unsupported cast op");
}

AffineMap updatePaddingMap(AffineMap origMap, const tile::PaddingInfo &padding,
                           MLIRContext *context) {
  assert(padding.lower.size() == origMap.getNumResults());
  SmallVector<AffineExpr, 4> newExprs;
  for (unsigned i = 0; i < origMap.getNumResults(); i++) {
    newExprs.push_back(origMap.getResult(i) + padding.lower[i]);
  }
  return simplifyAffineMap(
      AffineMap::get(origMap.getNumDims(), 0, newExprs, context));
}

static SmallVector<AffineExpr, 4>
getExprReplacements(ArrayRef<int64_t> lowBounds, MLIRContext *context) {
  SmallVector<AffineExpr, 4> replDims;
  for (unsigned i = 0; i < lowBounds.size(); ++i) {
    if (lowBounds[i] == 0) {
      replDims.emplace_back(getAffineDimExpr(i, context));
    } else {
      replDims.emplace_back(getAffineDimExpr(i, context) +
                            getAffineConstantExpr(lowBounds[i], context));
    }
  }
  return replDims;
}

AffineMap adjustMapByBounds(AffineMap origMap, ArrayRef<int64_t> lowBounds,
                            MLIRContext *context) {
  assert(lowBounds.size() == origMap.getNumDims());
  SmallVector<AffineExpr, 4> replDims = getExprReplacements(lowBounds, context);
  return simplifyAffineMap(origMap.replaceDimsAndSymbols(
      replDims, {}, origMap.getNumDims(), origMap.getNumSymbols()));
}

IntegerSet adjustConstraintsByBounds(IntegerSet origSet,
                                     ArrayRef<int64_t> lowBounds,
                                     MLIRContext *context) {
  assert(lowBounds.size() == origSet.getNumDims());
  SmallVector<AffineExpr, 4> replDims = getExprReplacements(lowBounds, context);
  return origSet.replaceDimsAndSymbols(replDims, {}, origSet.getNumDims(),
                                       origSet.getNumSymbols());
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
