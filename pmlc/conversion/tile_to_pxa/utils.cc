// Copyright 2021, Intel Corporation

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"

namespace pmlc::conversion::tile_to_pxa {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

using util::AggregationKind;

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
        // arith::ExtFOp: FloatType -> wider FloatType
        return builder.create<arith::ExtFOp>(loc, intoType, from).getResult();
      }
      // arith::TruncFOp: FloatType -> narrower FloatType
      return builder.create<arith::TruncFOp>(loc, intoType, from).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromSigned) {
        // arith::SIToFPOp: IntegerType -> FloatType
        return builder.create<arith::SIToFPOp>(loc, intoType, from).getResult();
      }
      // arith::UIToFPOp: IntegerType -> FloatType
      return builder.create<arith::UIToFPOp>(loc, intoType, from).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType i64Type = builder.getIntegerType(64);
      auto intCastOp = builder.create<arith::IndexCastOp>(loc, i64Type, from);
      return builder.create<arith::SIToFPOp>(loc, intoType, intCastOp)
          .getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (fromSigned) {
          // arith::ExtSIOp: IntegerType -> wider signed int
          return builder.create<arith::ExtSIOp>(loc, intoType, from)
              .getResult();
        }
        // arith::ExtUIOp: IntegerType -> wider unsigned int
        return builder.create<arith::ExtUIOp>(loc, intoType, from).getResult();
      }
      // arith::TruncIOp: IntegerType -> narrower IntegerType
      return builder.create<arith::TruncIOp>(loc, intoType, from).getResult();
    }
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (intoSigned) {
        // arith::FPToSIOp: FloatType -> signed IntegerType
        return builder.create<arith::FPToSIOp>(loc, intoType, from).getResult();
      }
      // arith::FPToUIOp: FloatType -> unsigned IntegerType
      return builder.create<arith::FPToUIOp>(loc, intoType, from).getResult();
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      IntegerType intType = builder.getIntegerType(intoIntType.getWidth());
      return builder.create<arith::IndexCastOp>(loc, intType, from);
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
  arith::AtomicRMWKind aggOp = arith::AtomicRMWKind::assign;
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

static RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    return rankedTensorType;
  }
  return RankedTensorType::get({}, type);
}

llvm::APFloat convertFloatUsingType(llvm::APFloat value, FloatType type) {
  bool losesInfo = false;
  value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return value;
}

Value createInit(OpBuilder &builder, Location loc, Type type,
                 AggregationKind agg) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (agg) {
    case AggregationKind::add: {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value,
                                                          floatType);
    }
    case AggregationKind::mul: {
      auto value = convertFloatUsingType(llvm::APFloat(1.0), floatType);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value,
                                                          floatType);
    }
    case AggregationKind::min: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), false);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value,
                                                          floatType);
    }
    case AggregationKind::max: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), true);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value,
                                                          floatType);
    }
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  } else if (auto intType = type.dyn_cast<IntegerType>()) {
    switch (agg) {
    case AggregationKind::add:
      return builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
    case AggregationKind::mul:
      return builder.create<mlir::arith::ConstantIntOp>(loc, 1, intType);
    case AggregationKind::min:
      return builder.create<mlir::arith::ConstantIntOp>(
          loc, std::numeric_limits<int>::max(), intType);
    case AggregationKind::max:
      return builder.create<mlir::arith::ConstantIntOp>(
          loc, std::numeric_limits<int>::min(), intType);
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  }
  llvm_unreachable("Unknown type for createInit");
}

Value buildBroadcastLoad(OpBuilder &builder, Location loc, Value operand,
                         unsigned outRank,
                         Optional<tile::PaddingInfo> maybePadding) {
  // Handle scalar values
  if (!operand.getType().isa<MemRefType>()) {
    return operand;
  }
  // handle broadcasts
  auto body = builder.getBlock();
  auto operandType = operand.getType().cast<MemRefType>();
  assert(operandType.getRank() <= outRank && "result rank < operand rank");
  ArrayRef<int64_t> shape = operandType.getShape();
  SmallVector<Value, 8> operandIdxs(operandType.getRank());
  for (unsigned i = 0; i < operandType.getRank(); i++) {
    unsigned j = outRank - i - 1;
    unsigned k = operandType.getRank() - i - 1;
    if (shape[k] == 1) {
      operandIdxs[k] = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    } else {
      operandIdxs[k] = body->getArgument(j);
    }
  }
  auto loadOp = builder.create<pxa::PxaLoadOp>(loc, operand, operandIdxs);
  if (maybePadding)
    updateAffineMap(loadOp, *maybePadding);
  return loadOp;
}

BufferAllocator::BufferAllocator(OpBuilder &builder, Operation *op,
                                 Type resultType) {
  // Gather some basic info
  TileToPXATypeConverter typeConverter;
  auto loc = op->getLoc();
  rankedTensorType = getRankedTensorType(resultType);
  elementType = typeConverter.convertType(rankedTensorType.getElementType());
  ArrayRef<int64_t> originalShape = rankedTensorType.getShape();
  auto shape = llvm::to_vector<8>(originalShape);

  // If padding is detected, expand the shape to accomodate.
  auto maybePadding = tile::getPaddingInfo(op);
  if (maybePadding) {
    for (unsigned i = 0, e = shape.size(); i < e; ++i) {
      shape[i] += maybePadding->lower[i] + maybePadding->upper[i];
    }
  }

  // Make an allocation for the output
  memRefType = MemRefType::get(shape, elementType);
  resultMemRef = builder.create<memref::AllocOp>(loc, memRefType);
  if (maybePadding) {
    auto initValue = createInit(builder, loc, elementType, maybePadding->agg);
    auto parallel = builder.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{memRefType},
        /*reductions=*/
        ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
        /*ranges=*/shape);
    auto parallelBuilder = parallel.getBodyBuilder();
    auto load =
        buildBroadcastLoad(parallelBuilder, loc, initValue, shape.size());
    auto stored =
        buildSimpleStore(parallelBuilder, loc, load, resultMemRef, llvm::None);
    parallelBuilder.create<AffineYieldOp>(loc, ValueRange{stored});
    resultMemRef = parallel.getResult(0);
  }
}

} // namespace pmlc::conversion::tile_to_pxa
