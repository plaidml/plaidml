// Copyright 2021, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/tile_to_linalg/pass_detail.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::tile_to_linalg {

namespace layer = dialect::layer;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

using util::AggregationKind;
using util::CombinationKind;

namespace {

static Type getElementType(Type type) {
  if (auto tensorType = type.dyn_cast<TensorType>()) {
    return tensorType.getElementType();
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return memRefType.getElementType();
  }
  return type;
}

static Type getElementType(Value value) {
  return getElementType(value.getType());
}

static llvm::APFloat convertFloatUsingType(llvm::APFloat value,
                                           FloatType type) {
  bool losesInfo = false;
  value.convert(type.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                &losesInfo);
  return value;
}

static FlatSymbolRefAttr createStubTraceFunc(ModuleOp module, StringAttr msg) {
  static unsigned idCounter = 0;
  auto uniqueId = idCounter++;
  auto symbol = llvm::formatv("__trace_{0}", uniqueId).str();
  auto context = module.getContext();
  OpBuilder builder(context);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = FunctionType::get(context, {}, {});
  auto funcOp = builder.create<FuncOp>(module.getLoc(), symbol, funcType,
                                       ArrayRef<NamedAttribute>{});
  funcOp->setAttr("msg", msg);
  funcOp->setAttr("trace", builder.getUnitAttr());
  funcOp->setAttr("id", builder.getI64IntegerAttr(uniqueId));
  funcOp.setPrivate();
  return SymbolRefAttr::get(context, symbol);
}

struct ConstantOpConversion : public OpConversionPattern<tile::ConstantOp> {
  using OpConversionPattern<tile::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdType = tile::toSignlessType(getElementType(op));
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
    tile::ContractionOpAdaptor adaptor(allOperands);
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
    tile::ContractionOpAdaptor adaptor(allOperands);
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
    bestType = tile::promoteTypes(bestType, type);
  }
  // Next, cast each operand to the 'final' type
  bool intoSigned = bestType.isSignedInteger();
  auto targetType = tile::toSignlessType(bestType);
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
    return op->getResult(0);
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
    Type boolType = rewriter.getI1Type();
    auto resultTypes = llvm::makeArrayRef(boolType);
    auto op = rewriter.create<OpType>(loc, resultTypes, promoted, attrs);
    return op->getResult(0);
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

static AffineMap
buildBroadcastMap(OpBuilder &builder, Location loc, Value operand,
                  unsigned outRank,
                  Optional<tile::PaddingInfo> maybePadding = llvm::None) {
  auto context = builder.getContext();
  // Handle scalar values
  if (!operand.getType().isa<RankedTensorType>()) {
    return AffineMap::get(outRank, 0, {}, context);
  }
  // handle broadcasts
  auto body = builder.getBlock();
  auto operandType = operand.getType().cast<RankedTensorType>();
  auto numDims = operandType.getRank();
  assert(numDims <= outRank && "result rank < operand rank");
  ArrayRef<int64_t> shape = operandType.getShape();
  SmallVector<AffineExpr, 4> exprs(numDims);
  for (unsigned i = 0; i < numDims; i++) {
    unsigned j = outRank - i - 1;
    unsigned k = numDims - i - 1;
    exprs[k] = (shape[k] == 1) ? builder.getAffineConstantExpr(0)
                               : builder.getAffineDimExpr(j);
  }
  auto map = AffineMap::get(outRank, 0, exprs, context);
  if (maybePadding) {
    map = updatePaddingMap(map, *maybePadding, context);
  }
  return map;
}

template <typename AggBuilder>
Value createAggOp(ConversionPatternRewriter &rewriter, Location loc,
                  Value aggValue, Value storedValue) {
  AggBuilder aggBuilder;
  return aggBuilder.create(
      rewriter, loc,
      /*resultType=*/aggValue.getType(),
      /*operands=*/ArrayRef<Value>{aggValue, storedValue},
      /*types=*/ArrayRef<Type>{aggValue.getType(), storedValue.getType()});
}

Value getAggResult(ConversionPatternRewriter &rewriter, Location loc,
                   AggregationKind agg, Value aggValue, Value storedValue) {
  // Do the reduction op
  Type aggType = aggValue.getType();
  switch (agg) {
  case AggregationKind::assign:
    return storedValue;
  case AggregationKind::add:
    if (aggType.isa<IntegerType>()) {
      return createAggOp<StdOp<AddIOp>>(rewriter, loc, aggValue, storedValue);
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<StdOp<AddFOp>>(rewriter, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::mul:
    if (aggType.isa<IntegerType>()) {
      return createAggOp<StdOp<MulIOp>>(rewriter, loc, aggValue, storedValue);
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<StdOp<MulFOp>>(rewriter, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::min:
    if (auto intType = aggType.dyn_cast<IntegerType>()) {
      if (intType.isSignedInteger()) {
        return createAggOp<CondOp<CmpIntOp<CmpIPredicate::slt>>>(
            rewriter, loc, aggValue, storedValue);
      } else {
        return createAggOp<CondOp<CmpIntOp<CmpIPredicate::ult>>>(
            rewriter, loc, aggValue, storedValue);
      }
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<CondOp<CmpFloatOp<CmpFPredicate::OLT>>>(
          rewriter, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::max:
    if (auto intType = aggType.dyn_cast<IntegerType>()) {
      if (intType.isSignedInteger()) {
        return createAggOp<CondOp<CmpIntOp<CmpIPredicate::sgt>>>(
            rewriter, loc, aggValue, storedValue);
      } else {
        return createAggOp<CondOp<CmpIntOp<CmpIPredicate::ugt>>>(
            rewriter, loc, aggValue, storedValue);
      }
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<CondOp<CmpFloatOp<CmpFPredicate::OGT>>>(
          rewriter, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  }
  llvm_unreachable("Invalid aggregation kind.");
}

struct BufferInitializer {
  Value resultTensor;
  RankedTensorType origTensorType;
  RankedTensorType newTensorType;
  Type elementType;

  BufferInitializer(OpBuilder &builder, Operation *op, Type resultType) {
    // Gather some basic info
    TileToLinalgTypeConverter typeConverter;
    auto loc = op->getLoc();
    origTensorType = resultType.cast<RankedTensorType>();
    elementType = typeConverter.convertType(origTensorType.getElementType());
    ArrayRef<int64_t> originalShape = origTensorType.getShape();
    auto shape = llvm::to_vector<8>(originalShape);

    // If padding is detected, expand the shape to accomodate.
    auto maybePadding = tile::getPaddingInfo(op);
    if (maybePadding) {
      for (unsigned i = 0, e = shape.size(); i < e; ++i) {
        shape[i] += maybePadding->lower[i] + maybePadding->upper[i];
      }
    }

    // Make an allocation for the output
    resultTensor =
        builder.create<linalg::InitTensorOp>(loc, shape, elementType);
    newTensorType = resultTensor.getType().cast<RankedTensorType>();

    if (maybePadding) {
      auto initValue = createInit(builder, loc, elementType, maybePadding->agg);
      auto fillOp = builder.create<linalg::FillOp>(loc,
                                                   /*value=*/initValue,
                                                   /*output=*/resultTensor);
      resultTensor = fillOp.getResult(0);
    }
  }
};

struct PrngOpConversion : public OpConversionPattern<tile::PrngOp> {
  using OpConversionPattern<tile::PrngOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PrngOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    op.emitError("Unsupported operation: tile::PrngOp.");
    return success();
  }
};

template <typename FromOpType, typename IntoOpBuilder,
          typename Matcher = AlwaysTrue>
struct EltwiseOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    Matcher pred;
    return pred(op);
  }

  void rewrite(FromOpType op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto context = op.getContext();
    BufferInitializer init(rewriter, op.getOperation(), op.result().getType());

    auto initType = init.newTensorType;
    auto shape = initType.getShape();
    auto numDims = shape.size();

    // Build indexing maps
    SmallVector<AffineMap, 4> idxMaps;
    SmallVector<Type, 4> argTypes;
    for (size_t i = 0; i < operands.size(); i++) {
      auto input = operands[i];
      auto maybePadding =
          tile::getPaddingInfo(op->getOperand(i).getDefiningOp());
      Type inputType = input.getType();
      auto idxMap = buildBroadcastMap(rewriter, loc, input, shape.size());
      if (maybePadding) {
        idxMap = updatePaddingMap(idxMap, *maybePadding, context);
      }
      auto shapedType = inputType.dyn_cast<ShapedType>();
      Type elementType = tile::toSignlessType(
          shapedType ? shapedType.getElementType() : inputType);
      idxMaps.emplace_back(idxMap);
      argTypes.emplace_back(elementType);
    }

    // For output indexing map and type
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    auto maybePadding = tile::getPaddingInfo(op);
    if (maybePadding) {
      outputMap = updatePaddingMap(outputMap, *maybePadding, context);
    }
    idxMaps.emplace_back(outputMap);
    argTypes.emplace_back(init.elementType);

    // Make a generic op
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{initType}, //
        /*inputs=*/operands,                       //
        /*outputs=*/ValueRange{init.resultTensor}, //
        /*indexingMaps=*/idxMaps,                  //
        /*iteratorTypes=*/SmallVector<StringRef, 4>(numDims, "parallel"));

    if (Attribute attr = op->getAttr("name")) {
      genericOp->setAttr("name", attr);
    }
    if (Attribute attr = op->getAttr("schedule")) {
      genericOp->setAttr("schedule", attr);
    }

    Block *body = rewriter.createBlock(&genericOp.region(),
                                       genericOp.region().begin(), argTypes);
    rewriter.setInsertionPointToStart(body);

    // Pop the output type
    argTypes.pop_back();
    // Create the standard op
    IntoOpBuilder intoOpBuilder;
    auto bodyArgs = body->getArguments();
    SmallVector<Value, 4> args(bodyArgs.begin(), bodyArgs.end() - 1);
    // Use the original operands' types to keep the (un)signed info
    SmallVector<Type, 4> operandTypes;
    for (auto type : op->getOperandTypes()) {
      operandTypes.push_back(getElementType(type));
    }

    Value result = intoOpBuilder.create(rewriter, loc, init.elementType, args,
                                        operandTypes);

    // Create the yield
    rewriter.create<linalg::YieldOp>(loc, ArrayRef<Value>{result});

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, genericOp.getResult(0));
  }
};

struct ContractionMapsAndShapes {
  // The contraction op contains the lower and upper bounds of the loop. Each
  // AffineMap in maps maps the loop dims to shape dims. The function makes
  // lower bounds be zero.
  ContractionMapsAndShapes(tile::ContractionOp op, ArrayRef<AffineMap> maps)
      : op(op), oldMaps(maps) {
    numDims = oldMaps[0].getNumDims();
    auto context = op.getContext();
    auto lowerBounds = op.lowerBounds().getValue();
    auto upperBounds = op.upperBounds().getValue();
    SmallVector<AffineExpr, 4> dimReplacements;
    for (unsigned i = 0; i < lowerBounds.getNumResults(); i++) {
      int64_t lowerBound, upperBound;
      if (auto lower =
              lowerBounds.getResult(i).dyn_cast<AffineConstantExpr>()) {
        lowerBound = lower.getValue();
      } else {
        op.emitError("Lower bound is not a constant.");
      }
      if (auto upper =
              upperBounds.getResult(i).dyn_cast<AffineConstantExpr>()) {
        upperBound = upper.getValue();
      } else {
        op.emitError("Upper bound is not a constant.");
      }
      shape.emplace_back(upperBound - lowerBound + 1);
      auto repl = getAffineDimExpr(i, context) +
                  getAffineConstantExpr(lowerBound, context);
      dimReplacements.emplace_back(simplifyAffineExpr(repl, numDims, 0));
    }
    for (auto oldMap : oldMaps) {
      auto newMap =
          oldMap.replaceDimsAndSymbols(dimReplacements, {}, numDims, 0);
      newMaps.emplace_back(simplifyAffineMap(newMap));
    }
  }

  // This function determines if all the loop dims appear as a single dim in
  // shape dims.
  bool shapeHasAllLoopDims() {
    llvm::SmallSet<unsigned, 4> dims;
    for (auto map : newMaps) {
      assert(numDims == map.getNumDims() &&
             "The input maps have different numbers of dimensions.");
      for (auto expr : map.getResults()) {
        if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
          dims.insert(dimExpr.getPosition());
        }
      }
    }
    return dims.size() == numDims;
  }

  tile::ContractionOp op;
  ArrayRef<AffineMap> oldMaps;
  SmallVector<AffineMap, 4> newMaps;
  SmallVector<int64_t, 4> shape;
  unsigned numDims;
};

template <CombinationKind comboKind, typename ComboBuilder,
          typename Matcher = AlwaysTrue>
struct ContractionOpConversion
    : public OpConversionPattern<tile::ContractionOp> {
  using OpConversionPattern<tile::ContractionOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const final {
    if (auto cionOp = dyn_cast<tile::ContractionOp>(op)) {
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

  void rewrite(tile::ContractionOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    try {
      tryRewrite(op, operands, rewriter);
    } catch (const std::exception &ex) {
      op.emitError(ex.what());
    }
  }

  void tryRewrite(tile::ContractionOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const {
    // Create an adaptor
    tile::ContractionOpAdaptor cionAdaptor(operands);
    auto cionOperands = cionAdaptor.operands();
    auto loc = op.getLoc();
    auto context = op.getContext();

    if (auto attr = op->getAttrOfType<StringAttr>("trace")) {
      auto module = op->getParentOfType<ModuleOp>();
      auto symbol = createStubTraceFunc(module, attr);
      rewriter.create<CallOp>(loc, symbol, ArrayRef<Type>{});
    }

    BufferInitializer bufInit(rewriter, op.getOperation(),
                              op.result().getType());
    auto initType = bufInit.newTensorType;
    auto initShape = initType.getShape();
    auto numInitDims = initShape.size();

    AffineMap identMap =
        AffineMap::getMultiDimIdentityMap(numInitDims, context);

    // Do initialization
    auto fillOp =
        rewriter.create<linalg::FillOp>(loc,
                                        /*value=*/cionAdaptor.init(),
                                        /*output=*/bufInit.resultTensor);
    auto filled = fillOp.getResult(0);

    // Prepare for indexing maps and iterator types
    SmallVector<AffineMap, 4> idxMaps;
    auto srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      auto src = srcs[i];
      auto map = src.cast<AffineMapAttr>().getValue();
      auto maybePadding =
          tile::getPaddingInfo(op.operands()[i].getDefiningOp());
      if (maybePadding) {
        map = updatePaddingMap(map, *maybePadding, context);
      }
      idxMaps.emplace_back(map);
    }
    auto sink = op.sink();
    auto maybePadding = tile::getPaddingInfo(op);
    if (maybePadding) {
      sink = updatePaddingMap(sink, *maybePadding, context);
    }
    idxMaps.emplace_back(sink);

    auto numDims = sink.getNumDims();
    SmallVector<StringRef, 4> iterTypes(numDims, "reduction");
    for (auto expr : sink.getResults()) {
      if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
        iterTypes[dimExpr.getPosition()] = "parallel";
      } else {
        auto dims = getUsedDims(expr);
        for (auto dim : dims) {
          iterTypes[dim] = "window";
        }
      }
    }

    // GenericOp requires that loop->shape maps can infer shape->loop maps.
    // However the inference function simply calls
    // AffineMap::inversePurmutation(loopToShapeMap) that is too simple to infer
    // shape->loop maps sometimes. So we can't get shape->loop maps and then
    // fail to verify GenericOp sometimes. In this case, we add a redundant
    // tensor and its AffineMap to specify the loop bound explicitly. Then it
    // can pass the GenericOp verification. Later, the redundant tensor could be
    // optimized becuase it is useless.
    ContractionMapsAndShapes info(op, idxMaps);
    bool needExtraMap = !info.shapeHasAllLoopDims();
    idxMaps = info.newMaps;
    SmallVector<Value, 4> inputs(cionOperands.begin(), cionOperands.end());
    if (needExtraMap) {
      // Create a redundant tensor with the dimensions of loop bounds
      auto extraTensor = rewriter.create<linalg::InitTensorOp>(
          loc, info.shape, bufInit.elementType);
      inputs.insert(inputs.begin(), extraTensor);
      AffineMap loopMap =
          AffineMap::getMultiDimIdentityMap(info.numDims, context);
      idxMaps.insert(idxMaps.begin(), loopMap);
    }

    // Create the main loop
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{initType}, //
        /*inputs=*/inputs,                         //
        /*outputs=*/ValueRange{filled},            //
        /*indexingMaps=*/idxMaps,                  //
        /*iteratorTypes=*/iterTypes);

    if (Attribute attr = op->getAttr("name"))
      genericOp->setAttr("name", attr);
    if (Attribute attr = op->getAttr("schedule"))
      genericOp->setAttr("schedule", attr);

    // Prepare for the body of GenericOp
    SmallVector<Type, 4> argTypes;
    if (needExtraMap) {
      argTypes.emplace_back(bufInit.elementType);
    }
    for (auto operand : cionOperands) {
      auto tensorType = operand.getType().cast<RankedTensorType>();
      argTypes.emplace_back(tensorType.getElementType());
    }
    argTypes.emplace_back(bufInit.elementType);
    Block *body = rewriter.createBlock(&genericOp.region(),
                                       genericOp.region().begin(), argTypes);
    rewriter.setInsertionPointToStart(body);

    // Do the combination op
    ComboBuilder comboBuilder;
    argTypes.pop_back();
    auto bodyArgs = body->getArguments();
    int offset = needExtraMap ? 1 : 0;
    SmallVector<Value, 2> inputArgs(bodyArgs.begin() + offset,
                                    bodyArgs.end() - 1);
    auto combined = comboBuilder.create(rewriter, loc, bufInit.elementType,
                                        inputArgs, argTypes);

    // Do the reduction op
    Value aggResult =
        getAggResult(rewriter, loc, op.agg(), bodyArgs.back(), combined);

    // Create the yield
    rewriter.create<linalg::YieldOp>(loc, ArrayRef<Value>{aggResult});

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, genericOp.getResult(0));
  }
};

struct IndexOpConversion : public OpConversionPattern<tile::IndexOp> {
  using OpConversionPattern<tile::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Gather some basic info
    auto loc = op.getLoc();
    auto context = op.getContext();
    TileToLinalgTypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType())
                          .cast<RankedTensorType>();
    auto elementType = resultType.getElementType().cast<IntegerType>();
    auto init = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), elementType);
    auto numDims = resultType.getRank();
    AffineMap identMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    SmallVector<StringRef, 4> iterTypes(numDims, "parallel");

    // Create a generic op to fill the result tensor
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{resultType},    //
        /*inputs=*/ValueRange{},                        //
        /*outputs=*/ValueRange{init.getResult()},       //
        /*indexingMaps=*/ArrayRef<AffineMap>{identMap}, //
        /*iteratorTypes=*/iterTypes);

    Block *body =
        rewriter.createBlock(&genericOp.region(), genericOp.region().begin(),
                             TypeRange{elementType});
    rewriter.setInsertionPointToStart(body);

    // Create index, index_cast and yield
    auto index =
        rewriter.create<linalg::IndexOp>(loc, op.axis().getZExtValue());
    auto cast = rewriter.create<mlir::IndexCastOp>(
        loc, index.getResult(),
        rewriter.getIntegerType(elementType.getWidth()));
    rewriter.create<linalg::YieldOp>(loc, ArrayRef<Value>{cast.getResult()});

    // Replace the op
    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

Optional<SmallVector<ReassociationIndices, 4>>
matchShape(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape) {
  SmallVector<ReassociationIndices, 4> result;
  int dstDim = dstShape.size() - 1;
  for (int srcDim = srcShape.size() - 1; srcDim >= 0; --srcDim) {
    int64_t size = dstShape[dstDim];
    ReassociationIndices dims = {dstDim};
    int startDstDim = dstDim - 1;
    while (startDstDim >= 0 && size < srcShape[srcDim]) {
      size *= dstShape[startDstDim];
      dims.insert(dims.begin(), startDstDim);
      --startDstDim;
    }
    if (size != srcShape[srcDim]) {
      return llvm::None;
    }
    dstDim = startDstDim;
    if (srcDim == 0) {
      for (int i = dstDim; i >= 0; --i) {
        if (dstShape[i] != 1) {
          return llvm::None;
        }
        dims.insert(dims.begin(), i);
      }
    }
    result.insert(result.begin(), dims);
  }
  return result;
}

struct ReshapeOpConversion : public OpConversionPattern<tile::ReshapeOp> {
  using OpConversionPattern<tile::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ReshapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor, to interpret the operands
    tile::ReshapeOpAdaptor adaptor(operands);

    auto tensor = adaptor.tensor();

    TileToLinalgTypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType());

    auto srcShape = tensor.getType().cast<RankedTensorType>().getShape();
    auto dstShape = resultType.cast<RankedTensorType>().getShape();

    if (auto dims = matchShape(dstShape, srcShape)) {
      rewriter.replaceOpWithNewOp<linalg::TensorCollapseShapeOp>(op, resultType,
                                                                 tensor, *dims);
    } else if (auto dims = matchShape(srcShape, dstShape)) {
      rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(op, resultType,
                                                               tensor, *dims);
    } else {
      // TODO: general reshape
      op.emitError("Reshape is not collapse or extend.");
    }
    return success();
  }
};

struct ShapeOpConversion : public OpConversionPattern<tile::ShapeOp> {
  using OpConversionPattern<tile::ShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ShapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor
    tile::ShapeOpAdaptor adaptor(operands);

    // Gather some basic info
    auto loc = op.getLoc();
    TileToLinalgTypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType())
                          .cast<RankedTensorType>();
    auto elementType = resultType.getElementType();

    auto shape64 =
        adaptor.tensor().getType().cast<RankedTensorType>().getShape();

    DenseIntElementsAttr shapeAttr;
    if (elementType.isInteger(32)) {
      SmallVector<int32_t, 4> shape32(shape64.begin(), shape64.end());
      shapeAttr = rewriter.getI32TensorAttr(shape32);
    } else if (elementType.isInteger(64)) {
      shapeAttr = rewriter.getI64TensorAttr(shape64);
    } else {
      op.emitError("Invalid return type of ShapeOp.");
    }
    rewriter.replaceOpWithNewOp<ConstantOp>(op, shapeAttr, resultType);
    return success();
  }
};

struct CastOpConversion : public OpConversionPattern<tile::CastOp> {
  using OpConversionPattern<tile::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::CastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto context = op.getContext();

    BufferInitializer init(rewriter, op.getOperation(), op.result().getType());
    auto initType = init.newTensorType;
    auto numDims = initType.getShape().size();
    auto inputMap =
        buildBroadcastMap(rewriter, loc, operands[0], initType.getRank());
    auto outputMap = AffineMap::getMultiDimIdentityMap(numDims, context);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{initType},                 //
        /*inputs=*/operands,                                       //
        /*outputs=*/ValueRange{init.resultTensor},                 //
        /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap}, //
        /*iteratorTypes=*/SmallVector<StringRef, 4>(numDims, "parallel"));

    auto originalSrcType = getElementType(op.tensor());
    auto convertedSrcType = getElementType(operands[0]);
    Block *body =
        rewriter.createBlock(&genericOp.region(), genericOp.region().begin(),
                             {convertedSrcType, init.elementType});
    rewriter.setInsertionPointToStart(body);

    // Create the standard cast op
    bool resultIsSigned =
        getElementType(op.result().getType()).isSignedInteger();
    auto result = createCastOp(rewriter, loc, body->getArgument(0),
                               originalSrcType.isSignedInteger(),
                               init.elementType, resultIsSigned);

    // Create the yield
    rewriter.create<linalg::YieldOp>(loc, ArrayRef<Value>{result});

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();

    // Convert the function signature
    TileToLinalgTypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    SmallVector<Type, 8> resultTypes;
    for (Type resultType : type.getResults()) {
      Type newResultType = typeConverter.convertType(resultType);
      if (!newResultType.isa<stdx::ArgpackType>()) {
        result.addInputs({newResultType});
      }
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(op.getContext(), result.getConvertedTypes(),
                                    resultTypes));

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
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
    return success();
  }
};

struct PragmaOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() == "trace") {
      return failure();
    }
    tile::PragmaOpAdaptor adaptor(operands);
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};

struct TraceOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() != "trace") {
      return failure();
    }
    tile::PragmaOpAdaptor adaptor(operands);
    auto module = op->getParentOfType<ModuleOp>();
    auto msg = op.attrs().getNamed("msg");
    if (!msg) {
      return failure();
    }
    auto symbol = createStubTraceFunc(module, msg->second.cast<StringAttr>());
    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{});
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};

struct ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    scf::ForOpAdaptor oldFor(operands);
    auto &oldBodyOps = op.getBody()->getOperations();
    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), oldFor.lowerBound(),
                                             oldFor.upperBound(), oldFor.step(),
                                             oldFor.initArgs());
    auto &newBodyOps = newOp.getBody()->getOperations();
    newBodyOps.splice(std::prev(newBodyOps.end()), oldBodyOps,
                      oldBodyOps.begin(), oldBodyOps.end());
    auto oldArgs = op.getBody()->getArguments();
    auto newArgs = newOp.getBody()->getArguments();
    for (unsigned i = 0; i < oldArgs.size(); ++i) {
      oldArgs[i].replaceAllUsesWith(newArgs[i]);
    }
    rewriter.replaceOp(op, newOp.results());
    return success();
  }
};

struct LowerTileToLinalgPass
    : public LowerTileToLinalgBase<LowerTileToLinalgPass> {
  void runOnOperation() final {
    // Inject tile.ident ops for each return operand that needs it.
    // argument, a constant value, or a reshape op.
    getOperation().walk([&](ReturnOp op) {
      OpBuilder builder(op);
      for (OpOperand &operand : op->getOpOperands()) {
        Value value = operand.get();
        bool needsIdent =                                  //
            value.isa<BlockArgument>() ||                  // Block arguemnt
            matchPattern(value, m_Constant()) ||           // Constant op
            matchPattern(value, m_Op<stdx::UnpackOp>()) || // Direct from unpack
            matchPattern(value, m_Op<tile::ReshapeOp>());  // Reshape op
        if (needsIdent) {
          Value copy = builder.create<tile::IdentOp>(op.getLoc(),
                                                     value.getType(), value);
          operand.set(copy);
        }
      }
    });

    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    TileToLinalgTypeConverter converter;
    target.addLegalDialect<mlir::AffineDialect,         //
                           mlir::linalg::LinalgDialect, //
                           mlir::StandardOpsDialect,    //
                           mlir::math::MathDialect,     //
                           mlir::memref::MemRefDialect, //
                           mlir::scf::SCFDialect,       //
                           layer::LayerDialect,         //
                           stdx::StdXDialect>();
    target.addLegalOp<scf::ForOp,   //
                      scf::YieldOp, //
                      scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp, //
                      ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<stdx::UnpackOp>([&](stdx::UnpackOp op) {
      return converter.isLegal(op.getResultTypes());
    });
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return converter.isLegal(op.getResultTypes()); });

    // Setup rewrite patterns
    using CmpIntLtOp =
        CmpIntInequalityOp<CmpIPredicate::slt, CmpIPredicate::ult>;
    using CmpIntLeOp =
        CmpIntInequalityOp<CmpIPredicate::sle, CmpIPredicate::ule>;
    using CmpIntGtOp =
        CmpIntInequalityOp<CmpIPredicate::sgt, CmpIPredicate::ugt>;
    using CmpIntGeOp =
        CmpIntInequalityOp<CmpIPredicate::sge, CmpIPredicate::uge>;
    RewritePatternSet patterns(&getContext());
    patterns.insert<
        CastOpConversion,     //
        ConstantOpConversion, //
        FuncOpConversion,     //
        IndexOpConversion,    //
        PragmaOpConversion,   //
        PrngOpConversion,     //
        ReshapeOpConversion,  //
        ReturnOpConversion,   //
        ShapeOpConversion,    //
        TraceOpConversion,    //
        ScfForOpConversion,   //
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
        EltwiseOpConversion<tile::ExpOp, StdOp<math::ExpOp>>,
        EltwiseOpConversion<tile::LogOp, StdOp<math::LogOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::PowOp, StdOp<stdx::PowOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ErfOp, StdOp<stdx::ErfOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CosOp, StdOp<math::CosOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::TanOp, StdOp<stdx::TanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::SinHOp, StdOp<stdx::SinHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CosHOp, StdOp<stdx::CosHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::SinOp, StdOp<math::SinOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::TanHOp, StdOp<math::TanhOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ACosOp, StdOp<stdx::ACosOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ASinOp, StdOp<stdx::ASinOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ATanOp, StdOp<stdx::ATanOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ACosHOp, StdOp<stdx::ACosHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ASinHOp, StdOp<stdx::ASinHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::ATanHOp, StdOp<stdx::ATanHOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::CeilOp, StdOp<mlir::CeilFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::FloorOp, StdOp<stdx::FloorOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::RoundOp, StdOp<stdx::RoundOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, StdOp<mlir::NegFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, NegIOp, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::AddFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::AddIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::SubFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::SubIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::MulFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::MulIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::DivFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::SignedDivIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::UnsignedDivIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::SqrtOp, StdOp<math::SqrtOp>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::RemFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::SignedRemIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::UnsignedRemIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpFloatOp<CmpFPredicate::OEQ>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpIntOp<CmpIPredicate::eq>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpFloatOp<CmpFPredicate::ONE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpIntOp<CmpIPredicate::ne>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpFloatOp<CmpFPredicate::OLT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpIntLtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpFloatOp<CmpFPredicate::OLE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpIntLeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpFloatOp<CmpFPredicate::OGT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpIntGtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpFloatOp<CmpFPredicate::OGE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpIntGeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::BitAndOp, StdOp<mlir::AndOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitOrOp, StdOp<mlir::OrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitNotOp, NotOp>,
        EltwiseOpConversion<tile::BitXorOp, StdOp<mlir::XOrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShlOp, StdOp<mlir::ShiftLeftOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::SignedShiftRightOp>,
                            FirstOperandIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::UnsignedShiftRightOp>,
                            FirstOperandIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::LogicalAndOp, LogicalOp<mlir::AndOp>>,
        EltwiseOpConversion<tile::LogicalNotOp, LogicalNotOp>,
        EltwiseOpConversion<tile::LogicalOrOp, LogicalOp<mlir::OrOp>>,
        EltwiseOpConversion<tile::LogicalXorOp, LogicalOp<mlir::XOrOp>>,
        EltwiseOpConversion<tile::ReluOp, StdOp<stdx::ReluOp>>,
        EltwiseOpConversion<tile::SelectOp, SelectOp>,
        EltwiseOpConversion<tile::IdentOp, FirstOperand>>(&getContext());

    populateTileToLinalgSpecialPatterns(patterns);

    // Run the conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileToLinalgPass() {
  return std::make_unique<LowerTileToLinalgPass>();
}

} // namespace pmlc::conversion::tile_to_linalg
