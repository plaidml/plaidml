// Copyright 2020, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/padding.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::tile_to_pxa {

namespace ew = dialect::eltwise;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

using namespace mlir; // NOLINT

using dialect::tile::AggregationKind;
using dialect::tile::CombinationKind;
using dialect::tile::ConstantOp;
using dialect::tile::ContractionOp;
using dialect::tile::ContractionOpAdaptor;
using dialect::tile::getPaddingInfo;
using dialect::tile::IndexOp;
using dialect::tile::PaddingInfo;
using dialect::tile::PrngOp;
using dialect::tile::ReshapeOp;
using dialect::tile::ReshapeOpAdaptor;
using dialect::tile::ShapeOp;
using dialect::tile::ShapeOpAdaptor;
using dialect::tile::TraceOp;

namespace {

struct TypeConverter : public mlir::TypeConverter {
  TypeConverter() {
    addConversion([](FunctionType type) { return type; });
    addConversion([](FloatType type) { return type; });
    addConversion([](IntegerType type) { return ew::toSignlessType(type); });
    addConversion([](MemRefType type) { return type; });
    addConversion([this](RankedTensorType type) {
      auto elementType = type.getElementType();
      auto newType = convertType(elementType);
      assert(newType && "could not convert type");
      return MemRefType::get(type.getShape(), newType);
    });
  }
};

static Type getElementType(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    return tensorType.getElementType();
  }
  return type;
}

static Type getElementType(Value value) {
  return getElementType(value.getType());
}

static RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    return rankedTensorType;
  }
  return RankedTensorType::get({}, type);
}

struct TileConstantOpConversion : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto value = op.getValue().cast<IntegerAttr>().getInt();
    rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(op, value);
    return success();
  }
};

static llvm::APFloat convertFloatUsingType(llvm::APFloat value,
                                           FloatType type) {
  bool losesInfo = false;
  value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return value;
}

struct ScalarConstantOpConversion
    : public OpConversionPattern<ew::ScalarConstantOp> {
  using OpConversionPattern<ew::ScalarConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ew::ScalarConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdType = ew::toSignlessType(getElementType(op));
    auto value = op.getValue();
    if (auto floatType = stdType.dyn_cast<FloatType>()) {
      auto floatAttr = value.cast<FloatAttr>();
      auto floatValue = convertFloatUsingType(floatAttr.getValue(), floatType);
      value = FloatAttr::get(floatType, floatValue);
    } else if (auto intType = stdType.dyn_cast<IntegerType>()) {
      auto intAttr = value.cast<IntegerAttr>();
      value = IntegerAttr::get(intType, intAttr.getInt());
    } else {
      llvm_unreachable("Invalid scalar constant op");
    }
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, value);
    return success();
  }
};

static Value createCastOp(OpBuilder &builder, Location loc, Value from,
                          bool fromSigned, Type intoType, bool intoSigned) {
  auto fromType = from.getType();
  if (fromType == intoType) {
    return from;
  }
  if (auto intoFloatType = intoType.dyn_cast<FloatType>()) {
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (fromFloatType.getWidth() < intoFloatType.getWidth()) {
        // FPExtOp: FloatType -> wider FloatType
        return builder.create<mlir::FPExtOp>(loc, from, intoType).getResult();
      }
      // FPTruncOp: FloatType -> narrower FloatType
      return builder.create<mlir::FPTruncOp>(loc, from, intoType).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromSigned) {
        // SIToFPOp: IntegerType -> FloatType
        return builder.create<mlir::SIToFPOp>(loc, from, intoType).getResult();
      } else {
        // UIToFPOp: IntegerType -> FloatType
        return builder.create<mlir::UIToFPOp>(loc, intoType, from).getResult();
      }
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      auto i64Type = builder.getIntegerType(64);
      auto intCastOp = builder.create<mlir::IndexCastOp>(loc, from, i64Type);
      return builder.create<mlir::SIToFPOp>(loc, intCastOp, intoType)
          .getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (fromSigned) {
          // SignExtendIOp: IntegerType -> wider signed int
          return builder.create<mlir::SignExtendIOp>(loc, from, intoType)
              .getResult();
        }
        // ZeroExtendIOp: IntegerType -> wider unsigned int
        return builder.create<mlir::ZeroExtendIOp>(loc, from, intoType)
            .getResult();
      }
      // TruncateIOp: IntegerType -> narrower IntegerType
      return builder.create<mlir::TruncateIOp>(loc, from, intoType).getResult();
    }
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (intoSigned) {
        // FPToSIOp: FloatType -> signed IntegerType
        return builder.create<mlir::FPToSIOp>(loc, from, intoType).getResult();
      } else {
        // FPToUIOp: FloatType -> unsigned IntegerType
        return builder.create<mlir::FPToUIOp>(loc, from, intoType).getResult();
      }
    }
    if (auto fromIndexType = fromType.dyn_cast<IndexType>()) {
      auto intType = builder.getIntegerType(intoIntType.getWidth());
      return builder.create<mlir::IndexCastOp>(loc, from, intType);
    }
  }
  llvm_unreachable("Unsupported cast op");
}

struct Matcher {
  LogicalResult operator()(Operation *op) { return success(match(op)); }
  virtual bool match(Operation *op) const { return false; }
};

struct AlwaysTrue : Matcher {
  bool match(Operation *op) const final { return true; }
};

template <typename InnerPredicate>
struct ResultIs : Matcher {
  bool match(Operation *op) const final {
    InnerPredicate pred;
    return pred.match(op->getResult(0).getType());
  }
};

template <typename InnerPredicate>
struct AnyOperandIs : Matcher {
  bool match(Operation *op) const final {
    for (auto operand : op->getOperands()) {
      InnerPredicate pred;
      if (pred.match(operand.getType())) {
        return true;
      }
    }
    return false;
  }
};

template <typename InnerPredicate>
struct OperandsAre : Matcher {
  bool match(Operation *op) const final {
    for (auto operand : op->getOperands()) {
      InnerPredicate pred;
      if (!pred.match(operand.getType())) {
        return false;
      }
    }
    return true;
  }
};

template <typename InnerPredicate>
struct FirstOperandIs : Matcher {
  bool match(Operation *op) const final {
    InnerPredicate pred;
    if (op->getNumOperands() == 0) {
      return false;
    }
    return pred.match(op->getOperand(0).getType());
  }
};

template <typename InnerPredicate>
struct AnyComparandIs : Matcher {
  bool match(Operation *op) const final {
    SmallVector<Value, 4> allOperands(op->getOperands());
    ContractionOpAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) ||
           pred.match(operands[1].getType());
  }
};

template <typename InnerPredicate>
struct ComparandsAre : Matcher {
  bool match(Operation *op) const final {
    SmallVector<Value, 4> allOperands(op->getOperands());
    ContractionOpAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) &&
           pred.match(operands[1].getType());
  }
};

template <typename InnerPredicate>
struct Not {
  bool match(Type type) const {
    InnerPredicate pred;
    return !pred.match(type);
  }
};

struct EltwiseFloat {
  bool match(Type type) const { return getElementType(type).isa<FloatType>(); }
};

struct EltwiseInteger {
  bool match(Type type) const {
    return getElementType(type).isa<IntegerType>();
  }
};

struct EltwiseSigned {
  bool match(Type type) const { return getElementType(type).isSignedInteger(); }
};

struct EltwiseUnsigned {
  bool match(Type type) const {
    return getElementType(type).isUnsignedInteger();
  }
};

struct FirstOperand {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    return operands.front();
  }
};

static Type promoteTypes(ConversionPatternRewriter &rewriter, Location loc,
                         ArrayRef<Value> operands, ArrayRef<Type> types,
                         SmallVectorImpl<Value> *into) {
  // First, determine the 'final' type that wins the promotion
  Type bestType;
  for (auto type : types) {
    bestType = ew::promoteTypes(bestType, type);
  }
  // Next, cast each operand to the 'final' type
  bool intoSigned = bestType.isSignedInteger();
  auto targetType = ew::toSignlessType(bestType);
  for (unsigned i = 0; i < operands.size(); i++) {
    auto dtype = types[i];
    auto operand = operands[i];
    auto castedValue =
        createCastOp(rewriter, loc, operand, dtype.isSignedInteger(),
                     targetType, intoSigned);
    into->push_back(castedValue);
  }
  return bestType;
}

struct NegIOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, resultType);
    auto neg = rewriter.create<mlir::SubIOp>(loc, zero, operands[0]);
    return neg.getResult();
  }
};

struct NotOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    // -(x + 1) = -1 - x
    auto negOne = rewriter.create<mlir::ConstantIntOp>(loc, -1, resultType);
    auto sub = rewriter.create<mlir::SubIOp>(loc, negOne, operands[0]);
    return sub.getResult();
  }
};

template <typename OpType>
struct StdOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    auto attrs = ArrayRef<NamedAttribute>{};
    auto resultTypes = llvm::makeArrayRef(resultType);
    auto op = rewriter.create<OpType>(loc, resultTypes, promoted, attrs);
    return op.getOperation()->getResult(0);
  }
};

struct SelectOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands.drop_front(), types.drop_front(),
                 &promoted);
    auto op = rewriter.create<mlir::SelectOp>(loc, operands[0], promoted[0],
                                              promoted[1]);
    return op.getResult();
  }
};

template <CmpFPredicate predicate>
struct CmpFloatOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::CmpFOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <CmpIPredicate predicate>
struct CmpIntOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <CmpIPredicate signedPred, CmpIPredicate unsignedPred>
struct CmpIntInequalityOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    auto bestType = promoteTypes(rewriter, loc, operands, types, &promoted);
    auto predicate = bestType.isSignedInteger() ? signedPred : unsignedPred;
    return rewriter
        .create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <typename OpType>
struct LogicalOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    for (unsigned i = 0; i < operands.size(); ++i) {
      auto &operand = operands[i];
      auto fromType = operand.getType();
      if (auto floatType = fromType.dyn_cast<FloatType>()) {
        auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
        auto zero =
            rewriter.create<mlir::ConstantFloatOp>(loc, value, floatType);
        promoted.push_back(
            rewriter
                .create<mlir::CmpFOp>(loc, CmpFPredicate::ONE, operand, zero)
                .getResult());
      } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
        auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, intType);
        promoted.push_back(
            rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::ne, operand, zero)
                .getResult());
      } else {
        llvm_unreachable("Unknown type for LogicalOp");
      }
    }
    auto attrs = ArrayRef<NamedAttribute>{};
    Type boolType = IntegerType::get(1, rewriter.getContext());
    auto resultTypes = llvm::makeArrayRef(boolType);
    auto op = rewriter.create<OpType>(loc, resultTypes, promoted, attrs);
    return op.getOperation()->getResult(0);
  }
};

struct LogicalNotOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    auto input = operands[0];
    auto fromType = input.getType();
    if (auto floatType = fromType.dyn_cast<FloatType>()) {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      auto zero = rewriter.create<mlir::ConstantFloatOp>(loc, value, floatType);
      return rewriter.create<mlir::CmpFOp>(loc, CmpFPredicate::OEQ, input, zero)
          .getResult();
    } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
      auto zero = rewriter.create<mlir::ConstantIntOp>(loc, 0, intType);
      return rewriter.create<mlir::CmpIOp>(loc, CmpIPredicate::eq, input, zero)
          .getResult();
    } else {
      llvm_unreachable("Unknown type for LogicalNotOp");
    }
  }
};

static Value createInit(OpBuilder &builder, Location loc, Type type,
                        AggregationKind agg) {
  if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (agg) {
    case AggregationKind::add: {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::mul: {
      auto value = convertFloatUsingType(llvm::APFloat(1.0), floatType);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::min: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), false);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::max: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), true);
      return builder.create<mlir::ConstantFloatOp>(loc, value, floatType);
    }
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  } else if (auto intType = type.dyn_cast<IntegerType>()) {
    switch (agg) {
    case AggregationKind::add:
      return builder.create<mlir::ConstantIntOp>(loc, 0, intType);
    case AggregationKind::mul:
      return builder.create<mlir::ConstantIntOp>(loc, 1, intType);
    case AggregationKind::min:
      return builder.create<mlir::ConstantIntOp>(
          loc, std::numeric_limits<int>::max(), intType);
    case AggregationKind::max:
      return builder.create<mlir::ConstantIntOp>(
          loc, std::numeric_limits<int>::min(), intType);
    default:
      llvm_unreachable("Unsupported aggregation for createInit");
    }
  }
  llvm_unreachable("Unknown type for createInit");
}

template <typename CmpOpBuilder>
struct CondOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(rewriter, loc, resultType,
                                   operands.take_front(2), types.take_front(2));
    auto zero = createInit(rewriter, loc, resultType, AggregationKind::add);
    return rewriter.create<mlir::SelectOp>(loc, cmp, operands[2], zero)
        .getResult();
  }
};

static void updateAffineMap(Operation *in, const PaddingInfo &padding) {
  auto accMap = in->getAttr("map").cast<AffineMapAttr>().getValue();
  assert(padding.lower.size() == accMap.getNumResults());
  SmallVector<AffineExpr, 4> newExprs;
  for (unsigned j = 0; j < accMap.getNumResults(); j++) {
    newExprs.push_back(accMap.getResult(j) + padding.lower[j]);
  }
  accMap = AffineMap::get(accMap.getNumDims(), 0, newExprs, in->getContext());
  in->setAttr("map", AffineMapAttr::get(accMap));
}

static Value
buildBroadcastLoad(OpBuilder &builder, Location loc, Value operand,
                   unsigned outRank,
                   Optional<PaddingInfo> maybePadding = llvm::None) {
  auto body = builder.getBlock();
  auto defOp = operand.getDefiningOp();
  Attribute attr;
  // Handle scalar values
  if (defOp && m_Constant(&attr).match(defOp)) {
    return operand;
  }
  // handle broadcasts
  auto operandType = operand.getType().cast<MemRefType>();
  assert(operandType.getRank() <= outRank && "result rank < operand rank");
  auto shape = operandType.getShape();
  SmallVector<Value, 8> operandIdxs(operandType.getRank());
  for (unsigned i = 0; i < operandType.getRank(); i++) {
    unsigned j = outRank - i - 1;
    unsigned k = operandType.getRank() - i - 1;
    if (shape[k] == 1) {
      operandIdxs[k] = builder.create<mlir::ConstantIndexOp>(loc, 0);
    } else {
      operandIdxs[k] = body->getArgument(j);
    }
  }
  auto loadOp = builder.create<pxa::PxaLoadOp>(loc, operand, operandIdxs);
  if (maybePadding)
    updateAffineMap(loadOp, *maybePadding);
  return loadOp;
}

static AtomicRMWKind convertAgg(AggregationKind agg, Type type) {
  switch (agg) {
  case AggregationKind::assign:
    return AtomicRMWKind::assign;
  case AggregationKind::add:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::addf;
    } else {
      return AtomicRMWKind::addi;
    }
  case AggregationKind::mul:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::mulf;
    } else {
      return AtomicRMWKind::muli;
    }
  case AggregationKind::min:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::minf;
    } else if (type.isSignedInteger()) {
      return AtomicRMWKind::mins;
    } else {
      return AtomicRMWKind::minu;
    }
  case AggregationKind::max:
    if (type.isa<FloatType>()) {
      return AtomicRMWKind::maxf;
    } else if (type.isSignedInteger()) {
      return AtomicRMWKind::maxs;
    } else {
      return AtomicRMWKind::maxu;
    }
  }
  llvm_unreachable("Invalid agg type in convertAgg");
}

static Value buildSimpleStore(OpBuilder &builder, Location loc, Value scalar,
                              Value memRef,
                              Optional<PaddingInfo> maybePadding) {
  auto body = builder.getBlock();
  auto memRefType = memRef.getType().cast<MemRefType>();
  auto elementType = memRefType.getElementType();
  if (elementType != scalar.getType()) {
    scalar = createCastOp(builder, loc, scalar, false, elementType, false);
  }
  auto aggOp = AtomicRMWKind::assign;
  auto idMap = builder.getMultiDimIdentityMap(memRefType.getRank());
  auto storeOp = builder.create<pxa::PxaReduceOp>(loc, aggOp, scalar, memRef,
                                                  idMap, body->getArguments());
  if (maybePadding)
    updateAffineMap(storeOp, *maybePadding);
  return storeOp;
}

struct BufferAllocator {
  Value resultMemRef;
  RankedTensorType rankedTensorType;
  MemRefType memRefType;
  Type elementType;

  BufferAllocator(OpBuilder &builder, Operation *op, Type resultType) {
    // Gather some basic info
    TypeConverter typeConverter;
    auto loc = op->getLoc();
    rankedTensorType = getRankedTensorType(resultType);
    elementType = typeConverter.convertType(rankedTensorType.getElementType());
    auto originalShape = rankedTensorType.getShape();
    auto shape = llvm::to_vector<8>(originalShape);

    // If padding is detected, expand the shape to accomodate.
    auto maybePadding = getPaddingInfo(op);
    if (maybePadding) {
      for (unsigned i = 0, e = shape.size(); i < e; ++i) {
        shape[i] += maybePadding->lower[i] + maybePadding->upper[i];
      }
    }

    // Make an allocation for the output
    memRefType = MemRefType::get(shape, elementType);
    resultMemRef = builder.create<AllocOp>(loc, memRefType);
    if (maybePadding) {
      auto initValue = createInit(builder, loc, elementType, maybePadding->agg);
      auto parallel = builder.create<AffineParallelOp>(
          loc,
          /*resultTypes=*/ArrayRef<Type>{memRefType},
          /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
          /*ranges=*/shape);
      auto parallelBuilder = parallel.getBodyBuilder();
      auto load =
          buildBroadcastLoad(parallelBuilder, loc, initValue, shape.size());
      auto stored = buildSimpleStore(parallelBuilder, loc, load, resultMemRef,
                                     llvm::None);
      parallelBuilder.create<AffineYieldOp>(loc, ValueRange{stored});
      resultMemRef = parallel.getResult(0);
    }
  }
};

struct PrngOpConversion : public OpConversionPattern<PrngOp> {
  using OpConversionPattern<PrngOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dialect::tile::PrngOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    PrngOp::Adaptor transformed(operands);
    auto loc = op.getLoc();
    BufferAllocator allocResult(rewriter, op.getOperation(),
                                op.result().getType());
    BufferAllocator stateResult(rewriter, op.getOperation(),
                                op.state().getType());
    auto resultType = allocResult.memRefType.getElementType();
    auto zero = createInit(rewriter, loc, resultType, AggregationKind::add);
    auto shape = allocResult.memRefType.getShape();
    auto parallel = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{allocResult.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/shape);
    auto parallelBuilder = parallel.getBodyBuilder();
    auto load = buildBroadcastLoad(parallelBuilder, loc, zero, shape.size());
    auto stored = buildSimpleStore(parallelBuilder, loc, load,
                                   allocResult.resultMemRef, llvm::None);
    parallelBuilder.create<AffineYieldOp>(loc, ValueRange{stored});
    SmallVector<Value, 2> x = {parallel.getResult(0), stateResult.resultMemRef};
    rewriter.replaceOp(op, x);

    return success();
  }
};

template <typename FromOpType, typename IntoOpBuilder,
          typename Matcher = AlwaysTrue>
struct EltwiseOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    IVLOG(2, "EltwiseOpConversion::match>");
    Matcher pred;
    return pred(op);
  }

  void rewrite(FromOpType op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/alloc.rankedTensorType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the loads
    SmallVector<Value, 4> scalars;
    for (size_t i = 0; i < operands.size(); i++) {
      auto maybePadding =
          getPaddingInfo(op.getOperation()->getOperand(i).getDefiningOp());
      scalars.push_back(buildBroadcastLoad(rewriter, loc, operands[i],
                                           alloc.memRefType.getRank(),
                                           maybePadding));
    }

    // Create the standard op
    SmallVector<Type, 4> operandTypes;
    for (auto type : op.getOperation()->getOperandTypes()) {
      operandTypes.push_back(getElementType(type));
    }
    IntoOpBuilder intoOpBuilder;
    auto result = intoOpBuilder.create(rewriter, loc, alloc.elementType,
                                       scalars, operandTypes);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, alloc.resultMemRef,
                                   getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, forOp.getResult(0));
  }
};

template <CombinationKind comboKind, typename ComboBuilder,
          typename Matcher = AlwaysTrue>
struct ContractionOpConversion : public OpConversionPattern<ContractionOp> {
  using OpConversionPattern<ContractionOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    IVLOG(2, "ContractionOpConversion::match>");
    if (auto cionOp = dyn_cast<ContractionOp>(op)) {
      if (cionOp.combo() != comboKind) {
        return failure();
      }
      if (!cionOp.lowerBounds().hasValue() ||
          !cionOp.upperBounds().hasValue()) {
        cionOp.emitError("contraction bounds must be computed");
        return failure();
      }
      Matcher pred;
      return pred(cionOp);
    }
    return failure();
  }

  void rewrite(ContractionOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    try {
      tryRewrite(op, operands, rewriter);
    } catch (const std::exception &ex) {
      op.emitError(ex.what());
    }
  }

  void tryRewrite(ContractionOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    // Create an adaptor
    ContractionOpAdaptor cionAdaptor(operands);
    auto cionOperands = cionAdaptor.operands();

    auto loc = op.getLoc();
    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Do initialization
    auto shape = alloc.rankedTensorType.getShape();
    auto parallel = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/shape);
    auto parallelBuilder = parallel.getBodyBuilder();
    auto maybePadding = getPaddingInfo(op.init().getDefiningOp());
    auto load = buildBroadcastLoad(parallelBuilder, loc, cionAdaptor.init(),
                                   shape.size(), maybePadding);
    auto store = buildSimpleStore(parallelBuilder, loc, load,
                                  alloc.resultMemRef, getPaddingInfo(op));
    if (maybePadding)
      updateAffineMap(store.getDefiningOp(), *maybePadding);
    parallelBuilder.create<AffineYieldOp>(loc, ValueRange{store});
    auto filled = parallel.getResult(0);

    // Determine lower and upper bounds.
    SmallVector<AffineExpr, 8> ubExprs;
    auto lowerBounds = op.lowerBounds().getValue();
    auto upperBounds = op.upperBounds().getValue();
    assert(lowerBounds.getNumResults() == upperBounds.getNumResults() &&
           "mismatched dims for lower and upper bounds");
    for (unsigned i = 0; i < lowerBounds.getNumResults(); i++) {
      auto ubExpr = upperBounds.getResult(i) + 1;
      auto upper = ubExpr.cast<AffineConstantExpr>().getValue();
      ubExprs.push_back(rewriter.getAffineConstantExpr(upper));
    }

    auto ubMap = AffineMap::get(0, 0, {ubExprs}, op.getContext());
    // Make the outer loops
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*lbMap=*/op.lowerBounds().getValue(),
        /*lbArgs=*/ArrayRef<Value>{},
        /*ubMap=*/ubMap,
        /*ubArgs=*/ArrayRef<Value>{});

    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    auto idxs = body->getArguments();

    // add constraints
    if (op.cons()) {
      auto cons = op.cons().getValue();
      auto ifOp = rewriter.create<AffineIfOp>(loc, TypeRange{alloc.memRefType},
                                              cons, idxs, true);
      rewriter.create<AffineYieldOp>(loc, ifOp.getOperation()->getResults());
      rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
      rewriter.create<AffineYieldOp>(loc, filled);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
    }

    // Create the loads + casts
    SmallVector<Value, 4> scalars;
    auto srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      auto operand = cionOperands[i];
      auto defOp = operand.getDefiningOp();
      Attribute attr;
      if (defOp && m_Constant(&attr).match(defOp)) {
        scalars.push_back(operand);
      } else {
        auto map = srcs[i].cast<AffineMapAttr>().getValue();
        auto loadOp = rewriter.create<pxa::PxaLoadOp>(loc, operand, map, idxs);
        auto maybePadding = getPaddingInfo(op.operands()[i].getDefiningOp());
        if (maybePadding)
          updateAffineMap(loadOp, *maybePadding);
        scalars.push_back(loadOp);
      }
    }

    // Do the combination op
    ComboBuilder comboBuilder;
    SmallVector<Type, 4> operandTypes;
    for (auto type : op.operands().getTypes()) {
      operandTypes.push_back(getElementType(type));
    }
    auto combined = comboBuilder.create(rewriter, loc, alloc.elementType,
                                        scalars, operandTypes);

    // Create the store
    auto resultMap = op.sink();
    pxa::PxaReduceOp reduceOp;
    auto agg = convertAgg(op.agg(), alloc.elementType);
    if (resultMap.isEmpty()) {
      SmallVector<Value, 0> emptyIdxs;
      reduceOp = rewriter.create<pxa::PxaReduceOp>(loc, agg, combined, filled,
                                                   resultMap, emptyIdxs);
    } else {
      reduceOp = rewriter.create<pxa::PxaReduceOp>(loc, agg, combined, filled,
                                                   resultMap, idxs);
    }
    maybePadding = getPaddingInfo(op);
    if (maybePadding)
      updateAffineMap(reduceOp, *maybePadding);
    rewriter.create<AffineYieldOp>(loc, ValueRange{reduceOp});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));
  }
};

struct IndexOpConversion : public OpConversionPattern<IndexOp> {
  using OpConversionPattern<IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "IndexOpConversion::matchAndRewrite>");

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{resultType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    auto idxs = body->getArguments();

    // Load the index value
    // TODO: add check that axis is within range in verifier
    auto axis = op.axis().getZExtValue();
    auto map = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)});
    auto apply = rewriter.create<mlir::AffineApplyOp>(loc, map, idxs[axis]);

    // Create the store
    auto cast = rewriter.create<mlir::IndexCastOp>(loc, apply,
                                                   rewriter.getIntegerType(32));
    auto stored =
        buildSimpleStore(rewriter, loc, cast, resultMemRef, getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReshapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "ReshapeOpConversion::matchAndRewrite>");

    // Create an adaptor, to interpret the operands
    ReshapeOpAdaptor adaptor(operands);

    auto tensor = adaptor.tensor();

    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType());

    rewriter.replaceOpWithNewOp<stdx::ReshapeOp>(op, resultType, tensor);
    return success();
  }
};

struct ShapeOpConversion : public OpConversionPattern<ShapeOp> {
  using OpConversionPattern<ShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "ShapeOpConversion::matchAndRewrite>");

    // Create an adaptor
    ShapeOpAdaptor adaptor(operands);

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto memRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Populate the buffer with the shape dims
    auto operandType = adaptor.tensor().getType().cast<MemRefType>();
    auto aggOp = AtomicRMWKind::assign;
    for (unsigned i = 0; i < operandType.getRank(); i++) {
      auto dim = rewriter.create<mlir::DimOp>(loc, adaptor.tensor(), i);
      auto cast = rewriter.create<mlir::IndexCastOp>(
          loc, dim, rewriter.getIntegerType(32));
      auto map = rewriter.getConstantAffineMap(i);
      memRef = rewriter.create<pxa::PxaReduceOp>(loc, aggOp, cast, memRef, map,
                                                 ArrayRef<Value>{});
    }

    // Replace the op
    rewriter.replaceOp(op, memRef);

    return success();
  }
};

struct CastOpConversion : public OpConversionPattern<ew::CastOp> {
  using OpConversionPattern<ew::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ew::CastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "CastOpConversion::matchAndRewrite>");

    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto oldResultType = op.result().getType();
    auto resultType =
        typeConverter.convertType(oldResultType).cast<MemRefType>();
    auto operand = operands[0];
    if (resultType == operand.getType()) {
      rewriter.replaceOp(op, operand);
      return success();
    }

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{resultType},
        /*reductions=*/ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign},
        /*ranges=*/resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the load
    auto scalar =
        buildBroadcastLoad(rewriter, loc, operand, resultType.getRank());

    // Create the standard cast op
    auto dtype = getElementType(op.tensor());
    bool resultIsSigned = getElementType(oldResultType).isSignedInteger();
    auto result = createCastOp(rewriter, loc, scalar, dtype.isSignedInteger(),
                               resultType.getElementType(), resultIsSigned);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, resultMemRef,
                                   getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();
    IVLOG(2, "FuncOpConversion::rewrite> " << debugString(type));

    // Convert the function signature
    TypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() +
                                                    type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    SmallVector<Type, 8> resultTypes;
    for (Type resultType : type.getResults()) {
      Type newResultType = typeConverter.convertType(resultType);
      result.addInputs({newResultType});
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(result.getConvertedTypes(), resultTypes,
                                    op.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);

    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    IVLOG(2, "ReturnOpConversion::matchAndRewrite>");
    auto &block = op.getParentRegion()->front();
    auto funcOp = op.getParentOfType<FuncOp>();
    auto blockArg = funcOp.getType().getNumInputs() - op.getNumOperands();
    for (auto operand : operands) {
      // Find very initial allocation of memref
      auto def = pxa::getIndirectDef(operand);
      def.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};

struct TraceOpConversion : public OpConversionPattern<TraceOp> {
  using OpConversionPattern<TraceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TraceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto module = op.getParentOfType<ModuleOp>();
    auto symbol = createStubFunc(module, op.msgAttr());
    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{});
    rewriter.replaceOp(op, op.in());
    return success();
  }

  FlatSymbolRefAttr createStubFunc(ModuleOp module, StringAttr msg) const {
    static unsigned idCounter = 0;
    auto uniqueId = idCounter++;
    auto symbol = llvm::formatv("__trace_{0}", uniqueId).str();
    auto context = module.getContext();
    OpBuilder builder(context);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType = FunctionType::get({}, {}, context);
    auto funcOp = builder.create<FuncOp>(module.getLoc(), symbol, funcType,
                                         ArrayRef<NamedAttribute>{});
    funcOp.setAttr("msg", msg);
    funcOp.setAttr("trace", builder.getUnitAttr());
    funcOp.setAttr("id", builder.getI64IntegerAttr(uniqueId));
    return SymbolRefAttr::get(symbol, context);
  }
};

struct LowerTileToPXAPass : public LowerTileToPXABase<LowerTileToPXAPass> {
  void runOnOperation() final {
    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    TypeConverter converter;
    target.addLegalDialect<mlir::AffineDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<dialect::pxa::PXADialect>();
    target.addLegalDialect<dialect::stdx::StdXDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp, ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op); });

    // Setup rewrite patterns
    using CmpIntLtOp =
        CmpIntInequalityOp<CmpIPredicate::slt, CmpIPredicate::ult>;
    using CmpIntLeOp =
        CmpIntInequalityOp<CmpIPredicate::sle, CmpIPredicate::ule>;
    using CmpIntGtOp =
        CmpIntInequalityOp<CmpIPredicate::sgt, CmpIPredicate::ugt>;
    using CmpIntGeOp =
        CmpIntInequalityOp<CmpIPredicate::sge, CmpIPredicate::uge>;
    OwningRewritePatternList patterns;
    patterns.insert<
        CastOpConversion,           //
        FuncOpConversion,           //
        IndexOpConversion,          //
        PrngOpConversion,           //
        ReshapeOpConversion,        //
        ReturnOpConversion,         //
        ScalarConstantOpConversion, //
        ShapeOpConversion,          //
        TileConstantOpConversion,   //
        TraceOpConversion,          //
        // TODO: SpecialOpConversion (GatherOp, ScatterOp, ZeroOp)
        ContractionOpConversion<CombinationKind::none, FirstOperand>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpFloatOp<CmpFPredicate::OEQ>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpIntOp<CmpIPredicate::eq>,
                                ComparandsAre<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpFloatOp<CmpFPredicate::OEQ>>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpIntOp<CmpIPredicate::eq>>,
                                AnyComparandIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::ExpOp, StdOp<mlir::ExpOp>>,
        EltwiseOpConversion<ew::LogOp, StdOp<mlir::LogOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::PowOp, StdOp<stdx::PowOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::ErfOp, StdOp<stdx::ErfOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::CosOp, StdOp<mlir::CosOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::TanOp, StdOp<stdx::TanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::SinHOp, StdOp<stdx::SinHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::CosHOp, StdOp<stdx::CosHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::SinOp, StdOp<mlir::SinOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::TanHOp, StdOp<mlir::TanhOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::ASinOp, StdOp<stdx::ASinOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::ACosOp, StdOp<stdx::ACosOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::ATanOp, StdOp<stdx::ATanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::CeilOp, StdOp<mlir::CeilFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::FloorOp, StdOp<stdx::FloorOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::RoundOp, StdOp<stdx::RoundOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<ew::NegOp, StdOp<mlir::NegFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::NegOp, NegIOp, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::AddOp, StdOp<mlir::AddFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::AddOp, StdOp<mlir::AddIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::SubOp, StdOp<mlir::SubFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::SubOp, StdOp<mlir::SubIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::MulOp, StdOp<mlir::MulFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::MulOp, StdOp<mlir::MulIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::DivFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::SignedDivIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::UnsignedDivIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::SqrtOp, StdOp<mlir::SqrtOp>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::RemFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::SignedRemIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::UnsignedRemIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::CmpEqOp, CmpFloatOp<CmpFPredicate::OEQ>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpEqOp, CmpIntOp<CmpIPredicate::eq>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpNeOp, CmpFloatOp<CmpFPredicate::ONE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpNeOp, CmpIntOp<CmpIPredicate::ne>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpLtOp, CmpFloatOp<CmpFPredicate::OLT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpLtOp, CmpIntLtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpLeOp, CmpFloatOp<CmpFPredicate::OLE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpLeOp, CmpIntLeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpGtOp, CmpFloatOp<CmpFPredicate::OGT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpGtOp, CmpIntGtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpGeOp, CmpFloatOp<CmpFPredicate::OGE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpGeOp, CmpIntGeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::BitAndOp, StdOp<mlir::AndOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitOrOp, StdOp<mlir::OrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitNotOp, NotOp>,
        EltwiseOpConversion<ew::BitXorOp, StdOp<mlir::XOrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitShlOp, StdOp<mlir::ShiftLeftOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitShrOp, StdOp<mlir::SignedShiftRightOp>,
                            FirstOperandIs<EltwiseSigned>>,
        EltwiseOpConversion<ew::BitShrOp, StdOp<mlir::UnsignedShiftRightOp>,
                            FirstOperandIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::LogicalAndOp, LogicalOp<mlir::AndOp>>,
        EltwiseOpConversion<ew::LogicalNotOp, LogicalNotOp>,
        EltwiseOpConversion<ew::LogicalOrOp, LogicalOp<mlir::OrOp>>,
        EltwiseOpConversion<ew::LogicalXorOp, LogicalOp<mlir::XOrOp>>,
        EltwiseOpConversion<ew::SelectOp, SelectOp>,
        EltwiseOpConversion<ew::IdentOp, FirstOperand>>(&getContext());
    // Run the conversion
    if (failed(applyFullConversion(getOperation(), target, patterns))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileToPXAPass() {
  return std::make_unique<LowerTileToPXAPass>();
}

} // namespace pmlc::conversion::tile_to_pxa
