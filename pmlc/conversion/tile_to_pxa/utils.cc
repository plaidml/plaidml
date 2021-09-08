// Copyright 2021, Intel Corporation

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"

namespace pmlc::conversion::tile_to_pxa {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

TileToPXATypeConverter::TileToPXATypeConverter() {
  addConversion([](FunctionType type) { return type; });
  addConversion([](FloatType type) { return type; });
  addConversion([](IntegerType type) { return tile::toSignlessType(type); });
  addConversion([](IndexType type) { return type; });
  addConversion([](MemRefType type) { return type; });
  addConversion([this](RankedTensorType type) {
    Type elementType = type.getElementType();
    Type newType = convertType(elementType);
    assert(newType && "could not convert type");
    return MemRefType::get(type.getShape(), newType);
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

Value buildSimpleStore(OpBuilder &builder, Location loc, Value scalar,
                       Value memRef, Optional<tile::PaddingInfo> maybePadding) {
  Block *body = builder.getBlock();
  auto memRefType = memRef.getType().cast<MemRefType>();
  Type elementType = memRefType.getElementType();
  if (elementType != scalar.getType()) {
    scalar = createCastOp(builder, loc, scalar, false, elementType, false);
  }
  AtomicRMWKind aggOp = AtomicRMWKind::assign;
  AffineMap idMap = builder.getMultiDimIdentityMap(memRefType.getRank());
  auto storeOp = builder.create<pxa::PxaReduceOp>(loc, aggOp, scalar, memRef,
                                                  idMap, body->getArguments());
  if (maybePadding)
    updateAffineMap(storeOp, *maybePadding);
  return storeOp;
}

void updateAffineMap(Operation *in, const tile::PaddingInfo &padding) {
  AffineMap accMap = in->getAttr("map").cast<AffineMapAttr>().getValue();
  assert(padding.lower.size() == accMap.getNumResults());
  SmallVector<AffineExpr, 4> newExprs;
  for (unsigned j = 0; j < accMap.getNumResults(); j++) {
    newExprs.push_back(accMap.getResult(j) + padding.lower[j]);
  }
  accMap = AffineMap::get(accMap.getNumDims(), 0, newExprs, in->getContext());
  in->setAttr("map", AffineMapAttr::get(accMap));
}

} // namespace pmlc::conversion::tile_to_pxa
