// Copyright 2021, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/tile_to_linalg/pass_detail.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
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
    TileToLinalgTypeConverter typeConverter;
    Type newType = typeConverter.convertType(op.getType());
    Type elementType = getElementType(newType);
    Attribute value = op.getValue();
    if (auto floatType = elementType.dyn_cast<FloatType>()) {
      auto floatAttr = value.cast<FloatAttr>();
      llvm::APFloat floatValue =
          convertFloatUsingType(floatAttr.getValue(), floatType);
      value = FloatAttr::get(floatType, floatValue);
    } else if (auto intType = elementType.dyn_cast<IntegerType>()) {
      auto intAttr = value.cast<IntegerAttr>();
      value = IntegerAttr::get(intType, intAttr.getInt());
    } else {
      llvm_unreachable("Invalid scalar constant op");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, elementType, value);
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
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    return operands.front();
  }
};

static Type promoteTypes(OpBuilder &builder, Location loc, ValueRange operands,
                         TypeRange types, SmallVectorImpl<Value> *into) {
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
    auto castedValue = createCastOp(
        builder, loc, operand, dtype.isSignedInteger(), targetType, intoSigned);
    into->push_back(castedValue);
  }
  return bestType;
}

struct NegIOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, resultType);
    auto neg = builder.create<mlir::arith::SubIOp>(loc, zero, operands[0]);
    return neg.getResult();
  }
};

struct NotOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    // -(x + 1) = -1 - x
    auto negOne = builder.create<mlir::arith::ConstantIntOp>(loc, -1, resultType);
    auto sub = builder.create<mlir::arith::SubIOp>(loc, negOne, operands[0]);
    return sub.getResult();
  }
};

template <typename OpType>
struct StdOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(builder, loc, operands, types, &promoted);
    ArrayRef<NamedAttribute> attrs;
    auto op =
        builder.create<OpType>(loc, TypeRange{resultType}, promoted, attrs);
    return op->getResult(0);
  }
};

struct SelectOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(builder, loc, operands.drop_front(), types.drop_front(),
                 &promoted);
    auto op = builder.create<mlir::SelectOp>(loc, operands[0], promoted[0],
                                             promoted[1]);
    return op.getResult();
  }
};

template <arith::CmpFPredicate predicate>
struct CmpFloatOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(builder, loc, operands, types, &promoted);
    return builder
        .create<mlir::arith::CmpFOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <arith::CmpIPredicate predicate>
struct CmpIntOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(builder, loc, operands, types, &promoted);
    return builder
        .create<mlir::arith::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <arith::CmpIPredicate signedPred, arith::CmpIPredicate unsignedPred>
struct CmpIntInequalityOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    auto bestType = promoteTypes(builder, loc, operands, types, &promoted);
    auto predicate = bestType.isSignedInteger() ? signedPred : unsignedPred;
    return builder
        .create<mlir::arith::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <typename OpType>
struct LogicalOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    SmallVector<Value, 2> promoted;
    for (Value operand : operands) {
      Type fromType = operand.getType();
      if (auto floatType = fromType.dyn_cast<FloatType>()) {
        llvm::APFloat value =
            convertFloatUsingType(llvm::APFloat(0.0), floatType);
        auto zero =
            builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
        promoted.push_back(
            builder.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, operand, zero)
                .getResult());
      } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
        auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
        promoted.push_back(
            builder.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ne, operand, zero)
                .getResult());
      } else {
        llvm_unreachable("Unknown type for LogicalOp");
      }
    }
    ArrayRef<NamedAttribute> attrs;
    Type boolType = builder.getI1Type();
    auto resultTypes = llvm::makeArrayRef(boolType);
    auto op = builder.create<OpType>(loc, resultTypes, promoted, attrs);
    return op->getResult(0);
  }
};

struct LogicalNotOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    auto input = operands[0];
    auto fromType = input.getType();
    if (auto floatType = fromType.dyn_cast<FloatType>()) {
      auto value = convertFloatUsingType(llvm::APFloat(0.0), floatType);
      auto zero = builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
      return builder.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, input, zero)
          .getResult();
    } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
      return builder.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, zero)
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
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::mul: {
      auto value = convertFloatUsingType(llvm::APFloat(1.0), floatType);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::min: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), false);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
    }
    case AggregationKind::max: {
      auto value = llvm::APFloat::getInf(floatType.getFloatSemantics(), true);
      return builder.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
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

template <typename CmpOpBuilder>
struct CondOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(builder, loc, resultType,
                                   operands.take_front(2), types.take_front(2));
    return builder.create<mlir::SelectOp>(loc, cmp, operands[0], operands[1])
        .getResult();
  }
};

template <typename CmpOpBuilder>
struct ContractionCondOp {
  Value create(OpBuilder &builder, Location loc, Type resultType,
               ValueRange operands, TypeRange types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(builder, loc, resultType,
                                   operands.take_front(2), types.take_front(2));
    auto zero = createInit(builder, loc, resultType, AggregationKind::add);
    return builder.create<mlir::SelectOp>(loc, cmp, operands[2], zero)
        .getResult();
  }
};

static AffineMap
buildBroadcastMap(OpBuilder &builder, Location loc, Value operand,
                  ShapedType outType,
                  Optional<tile::PaddingInfo> maybePadding = llvm::None) {
  MLIRContext *context = builder.getContext();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  unsigned outRank = outType.getRank();
  // Handle scalar values
  if (!operandType)
    return AffineMap::get(outRank, 0, context);

  // handle broadcasts
  Block *body = builder.getBlock();
  unsigned numDims = operandType.getRank();
  assert(numDims <= outRank && "result rank < operand rank");
  ArrayRef<int64_t> shape = operandType.getShape();
  ArrayRef<int64_t> outShape = outType.getShape();
  SmallVector<AffineExpr, 4> exprs(numDims);
  for (unsigned i = 0; i < numDims; i++) {
    unsigned j = outRank - i - 1;
    unsigned k = numDims - i - 1;
    exprs[k] = (shape[k] == 1 && outShape[j] != 1)
                   ? builder.getAffineConstantExpr(0)
                   : builder.getAffineDimExpr(j);
  }
  auto map = AffineMap::get(outRank, 0, exprs, context);
  if (maybePadding)
    map = updatePaddingMap(map, *maybePadding, context);
  return map;
}

template <typename AggBuilder>
Value createAggOp(OpBuilder &builder, Location loc, Value aggValue,
                  Value storedValue) {
  AggBuilder aggBuilder;
  return aggBuilder.create(
      builder, loc,
      /*resultType=*/aggValue.getType(),
      /*operands=*/ValueRange{aggValue, storedValue},
      /*types=*/TypeRange{aggValue.getType(), storedValue.getType()});
}

Value getAggResult(OpBuilder &builder, Location loc, AggregationKind agg,
                   Value aggValue, Value storedValue) {
  // Do the reduction op
  Type aggType = aggValue.getType();
  switch (agg) {
  case AggregationKind::assign:
    return storedValue;
  case AggregationKind::add:
    if (aggType.isa<IntegerType>()) {
      return createAggOp<StdOp<arith::AddIOp>>(builder, loc, aggValue, storedValue);
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<StdOp<arith::AddFOp>>(builder, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::mul:
    if (aggType.isa<IntegerType>()) {
      return createAggOp<StdOp<arith::MulIOp>>(builder, loc, aggValue, storedValue);
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<StdOp<arith::MulFOp>>(builder, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::min:
    if (auto intType = aggType.dyn_cast<IntegerType>()) {
      if (intType.isSignedInteger()) {
        return createAggOp<CondOp<CmpIntOp<arith::CmpIPredicate::slt>>>(
            builder, loc, aggValue, storedValue);
      } else {
        return createAggOp<CondOp<CmpIntOp<arith::CmpIPredicate::ult>>>(
            builder, loc, aggValue, storedValue);
      }
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<CondOp<CmpFloatOp<arith::CmpFPredicate::OLT>>>(
          builder, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  case AggregationKind::max:
    if (auto intType = aggType.dyn_cast<IntegerType>()) {
      if (intType.isSignedInteger()) {
        return createAggOp<CondOp<CmpIntOp<arith::CmpIPredicate::sgt>>>(
            builder, loc, aggValue, storedValue);
      } else {
        return createAggOp<CondOp<CmpIntOp<arith::CmpIPredicate::ugt>>>(
            builder, loc, aggValue, storedValue);
      }
    } else if (aggType.isa<FloatType>()) {
      return createAggOp<CondOp<CmpFloatOp<arith::CmpFPredicate::OGT>>>(
          builder, loc, aggValue, storedValue);
    } else {
      llvm_unreachable("Invalid aggregation value type.");
    }
  }
  llvm_unreachable("Invalid aggregation kind.");
}

struct TensorInitializer {
  Value resultTensor;

  TensorInitializer(OpBuilder &builder, Operation *op, Type resultType,
                    bool padding = true) {
    // Gather some basic info
    TileToLinalgTypeConverter typeConverter;
    RankedTensorType oldTensorType = resultType.cast<RankedTensorType>();
    Type elementType =
        typeConverter.convertType(oldTensorType.getElementType());
    SmallVector<int64_t, 8> shape =
        llvm::to_vector<8>(oldTensorType.getShape());

    // If padding is detected, expand the shape to accomodate.
    Optional<tile::PaddingInfo> maybePadding = tile::getPaddingInfo(op);
    if (padding && maybePadding) {
      for (unsigned i = 0, e = shape.size(); i < e; ++i) {
        shape[i] += maybePadding->lower[i] + maybePadding->upper[i];
      }
    }

    // Make an allocation for the output
    Location loc = op->getLoc();
    resultTensor =
        builder.create<linalg::InitTensorOp>(loc, shape, elementType);

    if (padding && maybePadding) {
      Value initValue =
          createInit(builder, loc, elementType, maybePadding->agg);
      auto fillOp = builder.create<linalg::FillOp>(loc,
                                                   /*value=*/initValue,
                                                   /*output=*/resultTensor);
      resultTensor = fillOp.getResult(0);
    }
  }

  RankedTensorType getType() {
    return resultTensor.getType().cast<RankedTensorType>();
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

  void rewrite(FromOpType op, typename FromOpType::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    TensorInitializer init(rewriter, op, op.result().getType(),
                           /*padding=*/false);

    RankedTensorType initType = init.getType();
    unsigned numDims = initType.getRank();

    // Build indexing maps
    SmallVector<AffineMap, 4> idxMaps;
    for (size_t i = 0; i < operands.size(); i++) {
      Optional<tile::PaddingInfo> maybePadding =
          tile::getPaddingInfo(op->getOperand(i).getDefiningOp());
      AffineMap idxMap =
          buildBroadcastMap(rewriter, loc, operands[i], initType, maybePadding);
      idxMaps.emplace_back(idxMap);
    }

    // For output indexing map and type
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(numDims, context);
    idxMaps.emplace_back(outputMap);

    // Make a generic op
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{initType},
        /*inputs=*/operands,
        /*outputs=*/ValueRange{init.resultTensor},
        /*indexingMaps=*/idxMaps,
        /*iteratorTypes=*/
        SmallVector<StringRef, 4>(numDims, getParallelIteratorTypeName()),
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          IntoOpBuilder intoOpBuilder;
          ValueRange inputs = args.drop_back();
          // Use the original operand types to retain signedness.
          SmallVector<Type, 4> inputTypes = llvm::to_vector<4>(
              llvm::map_range(op->getOperandTypes(),
                              [](Type type) { return getElementType(type); }));
          Value result = intoOpBuilder.create(
              builder, loc, initType.getElementType(), inputs, inputTypes);
          builder.create<linalg::YieldOp>(loc, ValueRange{result});
        });

    if (Attribute attr = op->getAttr("name"))
      genericOp->setAttr("name", attr);
    if (Attribute attr = op->getAttr("schedule"))
      genericOp->setAttr("schedule", attr);

    Value outTensor = genericOp.getResult(0);

    // If we need to pad the result
    if (Optional<tile::PaddingInfo> maybePadding = tile::getPaddingInfo(op)) {
      Value initValue = createInit(rewriter, loc, initType.getElementType(),
                                   maybePadding->agg);
      auto pad = rewriter.create<linalg::PadTensorOp>(
          loc,
          /*source=*/outTensor,
          /*staticLow=*/maybePadding->lower,
          /*staticHigh=*/maybePadding->upper,
          /*low=*/ValueRange{},
          /*high=*/ValueRange{});
      SmallVector<Type, 4> padArgs(numDims, rewriter.getIndexType());
      OpBuilder::InsertionGuard guard(rewriter);
      Block *padBody =
          rewriter.createBlock(&pad.region(), pad.region().begin(), padArgs);
      rewriter.create<linalg::YieldOp>(loc, ValueRange{initValue});
      outTensor = pad.getResult();
    }

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, outTensor);
  }
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

  void rewrite(tile::ContractionOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const final {
    MLIRContext *context = op.getContext();
    Location loc = op.getLoc();
    ValueRange cionOperands = adaptor.getOperands();

    TensorInitializer init(rewriter, op, op.result().getType(),
                           /*padding=*/false);
    RankedTensorType resultType = init.getType();
    unsigned numDims = resultType.getRank();

    // Do initialization
    Value initValue = adaptor.init();

    // Broadcast the initializer as necessary.
    if (auto tensorType = initValue.getType().dyn_cast<RankedTensorType>()) {
      if (tensorType != resultType) {
        AffineMap inputMap =
            buildBroadcastMap(rewriter, loc, initValue, resultType);
        AffineMap outputMap =
            AffineMap::getMultiDimIdentityMap(numDims, context);
        auto broadcastOp = rewriter.create<linalg::GenericOp>(
            loc,
            /*resultTensorTypes=*/TypeRange{resultType},
            /*inputs=*/ValueRange{initValue},
            /*outputs=*/ValueRange{init.resultTensor},
            /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap},
            /*iteratorTypes=*/
            SmallVector<StringRef, 4>(numDims, getParallelIteratorTypeName()),
            /*doc=*/"",
            /*libraryCall=*/"",
            [&](OpBuilder &builder, Location loc, ValueRange args) {
              builder.create<linalg::YieldOp>(loc, args.take_front());
            });
        initValue = broadcastOp.getResult(0);
      }
    } else {
      // Deal with scalar initializer values.
      auto fillOp =
          rewriter.create<linalg::FillOp>(loc,
                                          /*value=*/initValue,
                                          /*output=*/init.resultTensor);
      initValue = fillOp.result();
    }

    Optional<tile::PaddingInfo> maybeOpPadding = tile::getPaddingInfo(op);
    if (maybeOpPadding) {
      Value exteriorValue = createInit(
          rewriter, loc, resultType.getElementType(), maybeOpPadding->agg);
      auto pad = rewriter.create<linalg::PadTensorOp>(
          loc,
          /*source=*/initValue,
          /*staticLow=*/maybeOpPadding->lower,
          /*staticHigh=*/maybeOpPadding->upper,
          /*low=*/ValueRange{},
          /*high=*/ValueRange{});
      SmallVector<Type, 4> padArgs(numDims, rewriter.getIndexType());
      OpBuilder::InsertionGuard guard(rewriter);
      Block *padBody =
          rewriter.createBlock(&pad.region(), pad.region().begin(), padArgs);
      rewriter.create<linalg::YieldOp>(loc, ValueRange{exteriorValue});
      initValue = pad.getResult();
      resultType = initValue.getType().cast<RankedTensorType>();
    }

    auto lowMap = op.lowerBounds().getValue();
    bool zeroLowBounds = true;
    SmallVector<int64_t, 4> lowBounds;
    for (auto expr : lowMap.getResults()) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        int64_t low = constExpr.getValue();
        lowBounds.emplace_back(low);
        zeroLowBounds &= (low == 0);
      } else {
        op.emitError("Non-constant bound for contraction.");
      }
    }

    // Prepare for indexing maps and iterator types
    SmallVector<AffineMap, 4> idxMaps;
    ArrayRef<Attribute> srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      AffineMap map = srcs[i].cast<AffineMapAttr>().getValue();
      if (Optional<tile::PaddingInfo> maybeOperandPadding =
              tile::getPaddingInfo(op.operands()[i].getDefiningOp())) {
        map = updatePaddingMap(map, *maybeOperandPadding, context);
      }
      if (!zeroLowBounds) {
        map = adjustMapByBounds(map, lowBounds, context);
      }
      idxMaps.emplace_back(map);
    }
    AffineMap sink = op.sink();
    if (maybeOpPadding) {
      sink = updatePaddingMap(sink, *maybeOpPadding, context);
    }
    if (!zeroLowBounds) {
      sink = adjustMapByBounds(sink, lowBounds, context);
    }
    idxMaps.emplace_back(sink);

    SmallVector<StringRef, 4> iterTypes(sink.getNumDims(),
                                        getReductionIteratorTypeName());
    for (AffineExpr expr : sink.getResults()) {
      if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
        iterTypes[dimExpr.getPosition()] = getParallelIteratorTypeName();
      } else {
        for (int64_t dim : getUsedDims(expr)) {
          iterTypes[dim] = getWindowIteratorTypeName();
        }
      }
    }

    SmallVector<int64_t, 8> iterRanges;
    SmallVector<int64_t> lowerBounds =
        op.lowerBounds().getValue().getConstantResults();
    SmallVector<int64_t> upperBounds =
        op.upperBounds().getValue().getConstantResults();
    for (unsigned i = 0; i < lowerBounds.size(); i++) {
      iterRanges.emplace_back(upperBounds[i] - lowerBounds[i] + 1);
    }

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/cionOperands,
        /*outputs=*/initValue,
        /*indexingMaps=*/idxMaps,
        /*iteratorTypes=*/iterTypes,
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          ComboBuilder comboBuilder;
          ValueRange comboArgs = args.drop_back();
          Value combined =
              comboBuilder.create(builder, loc, args.back().getType(),
                                  comboArgs, comboArgs.getTypes());
          Value aggregate =
              getAggResult(builder, loc, op.agg(), args.back(), combined);
          builder.create<linalg::YieldOp>(loc, ValueRange{aggregate});
        });
    genericOp->setAttr(getIteratorRangesAttrName(),
                       rewriter.getI64ArrayAttr(iterRanges));
    if (Attribute attr = op->getAttr("name"))
      genericOp->setAttr("name", attr);
    if (Attribute attr = op->getAttr("schedule"))
      genericOp->setAttr("schedule", attr);

    // add constraints
    if (op.cons()) {
      auto cons = op.cons().getValue();
      if (!zeroLowBounds) {
        cons = adjustConstraintsByBounds(cons, lowBounds, context);
      }
      genericOp->setAttr("constraints", IntegerSetAttr::get(cons));
    }

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, genericOp.getResult(0));
  }
};

struct IndexOpConversion : public OpConversionPattern<tile::IndexOp> {
  using OpConversionPattern<tile::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Gather some basic info
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();
    TileToLinalgTypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType())
                          .cast<RankedTensorType>();
    auto init = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    unsigned numDims = resultType.getRank();
    AffineMap identMap = AffineMap::getMultiDimIdentityMap(numDims, context);

    // Create a generic op to fill the result tensor
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{resultType},
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{init.getResult()},
        /*indexingMaps=*/ArrayRef<AffineMap>{identMap},
        /*iteratorTypes=*/
        SmallVector<StringRef, 4>(numDims, getParallelIteratorTypeName()),
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          auto index =
              builder.create<linalg::IndexOp>(loc, op.axis().getZExtValue());
          auto cast = builder.create<arith::IndexCastOp>(loc, index.getResult(),
                                                  resultType.getElementType());
          builder.create<linalg::YieldOp>(loc, ValueRange{cast.getResult()});
        });

    // Replace the op
    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

Optional<SmallVector<ReassociationIndices, 4>>
matchShape(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape) {
  SmallVector<ReassociationIndices, 4> result;
  if (dstShape.empty()) {
    if (srcShape.empty()) {
      return result;
    }
    return llvm::None;
  }
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
  matchAndRewrite(tile::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
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
      // General reshape. Collapse the tensor into 1-D and then expand it to the
      // result shape.
      int64_t size = 1;
      ReassociationIndices collapseDims;
      for (unsigned i = 0; i < srcShape.size(); ++i) {
        size *= srcShape[i];
        collapseDims.emplace_back(i);
      }
      auto tmpType = RankedTensorType::get(
          ArrayRef{size}, resultType.cast<RankedTensorType>().getElementType());
      auto collapse = rewriter.create<linalg::TensorCollapseShapeOp>(
          op.getLoc(), tmpType, tensor, collapseDims);
      ReassociationIndices expandDims;
      for (unsigned i = 0; i < dstShape.size(); ++i) {
        expandDims.emplace_back(i);
      }
      rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
          op, resultType, collapse.getResult(), expandDims);
    }
    return success();
  }
};

struct ShapeOpConversion : public OpConversionPattern<tile::ShapeOp> {
  using OpConversionPattern<tile::ShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
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
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, shapeAttr, resultType);
    return success();
  }
};

struct CastOpConversion : public OpConversionPattern<tile::CastOp> {
  using OpConversionPattern<tile::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    TensorInitializer init(rewriter, op, op.result().getType());
    RankedTensorType initType = init.getType();
    unsigned numDims = initType.getRank();
    AffineMap inputMap =
        buildBroadcastMap(rewriter, loc, adaptor.getOperands()[0], initType);
    AffineMap outputMap = AffineMap::getMultiDimIdentityMap(numDims, context);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/TypeRange{initType},
        /*inputs=*/adaptor.getOperands(),
        /*outputs=*/ValueRange{init.resultTensor},
        /*indexingMaps=*/ArrayRef<AffineMap>{inputMap, outputMap},
        /*iteratorTypes=*/
        SmallVector<StringRef, 4>(numDims, getParallelIteratorTypeName()),
        /*doc=*/"",
        /*libraryCall=*/"",
        [&](OpBuilder &builder, Location loc, ValueRange args) {
          Type originalSrcType = getElementType(op.tensor());
          Type convertedSrcType = getElementType(adaptor.getOperands()[0]);
          // Create the standard cast op
          bool resultIsSigned =
              getElementType(op.result().getType()).isSignedInteger();
          auto result = createCastOp(rewriter, loc, args[0],
                                     originalSrcType.isSignedInteger(),
                                     initType.getElementType(), resultIsSigned);
          rewriter.create<linalg::YieldOp>(loc, ValueRange{result});
        });

    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

template <typename FuncLikeOp>
struct FuncOpConversion : public OpConversionPattern<FuncLikeOp> {
  using OpConversionPattern<FuncLikeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncLikeOp op, OpAdaptor adaptor,
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
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
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
  matchAndRewrite(ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct PragmaOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() == "trace") {
      return failure();
    }
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};

template <typename SpecialOp>
struct SpecialOpConversion : public OpConversionPattern<SpecialOp> {
  using OpConversionPattern<SpecialOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SpecialOp op, typename SpecialOp::Adaptor adaptor
                  ConversionPatternRewriter &rewriter) const final {
    TileToLinalgTypeConverter typeConverter;
    SmallVector<Type> resultTypes;
    (void)typeConverter.convertTypes(op->getResultTypes(), resultTypes);
    rewriter.replaceOpWithNewOp<SpecialOp>(op, resultTypes, adaptor.getOperands(),
                                           op->getAttrs());
    return success();
  }
};

struct TraceOpConversion : public OpConversionPattern<tile::PragmaOp> {
  using OpConversionPattern<tile::PragmaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PragmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (op.op() != "trace") {
      return failure();
    }
    auto module = op->getParentOfType<ModuleOp>();
    auto msg = op.attrs().getNamed("msg");
    if (!msg) {
      return failure();
    }
    auto symbol = createStubTraceFunc(module, msg->second.cast<StringAttr>());
    rewriter.create<CallOp>(op.getLoc(), symbol, TypeRange{});
    rewriter.replaceOp(op, adaptor.tensor());
    return success();
  }
};

struct ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor oldFor,
                  ConversionPatternRewriter &rewriter) const final {
    auto &oldBodyOps = op.getBody()->getOperations();
    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), oldFor.getLowerBound(),
                                             oldFor.getUpperBound(), oldFor.getStep(),
                                             oldFor.getInitArgs());
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
    auto module = getOperation();
    // Inject tile.ident ops for each return operand that needs it.
    // argument, a constant value, or a reshape op.
    auto injectIdent = [](Operation *op) {
      OpBuilder builder(op);
      for (OpOperand &operand : op->getOpOperands()) {
        Value value = operand.get();
        Value def = dialect::pxa::getIndirectDef(value);
        bool needsIdent =                                   //
            value.isa<BlockArgument>() ||                   // Block arguemnt
            matchPattern(value, m_Constant()) ||            // Constant op
            matchPattern(value, m_Op<tile::ReshapeOp>()) || // Reshape op
            def.getParentRegion() != op->getParentRegion();
        if (needsIdent) {
          Value copy = builder.create<tile::IdentOp>(op->getLoc(),
                                                     value.getType(), value);
          operand.set(copy);
        }
      }
    };
    module.walk([&](ReturnOp op) { injectIdent(op); });
    module.walk([&](stdx::YieldOp op) { injectIdent(op); });

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
                           arith::ArithmeticDialect,
                           stdx::StdXDialect>();
    target.addLegalOp<scf::ForOp,   //
                      scf::YieldOp, //
                      scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp, //
                      ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<stdx::ClosureOp>([&](stdx::ClosureOp op) {
      return converter.isSignatureLegal(op.getType());
    });
    target.addDynamicallyLegalOp<ReturnOp>(
        [&](ReturnOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return converter.isLegal(op.getResultTypes()); });

    target.addDynamicallyLegalOp<tile::ArgSortOp>(
        [&](tile::ArgSortOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<tile::GatherOp>(
        [&](tile::GatherOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<tile::PrngOp>(
        [&](tile::PrngOp op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<tile::ScatterOp>(
        [&](tile::ScatterOp op) { return converter.isLegal(op); });

    // Setup rewrite patterns
    using CmpIntLtOp =
        CmpIntInequalityOp<arith::CmpIPredicate::slt, arith::CmpIPredicate::ult>;
    using CmpIntLeOp =
        CmpIntInequalityOp<arith::CmpIPredicate::sle, arith::CmpIPredicate::ule>;
    using CmpIntGtOp =
        CmpIntInequalityOp<arith::CmpIPredicate::sgt, arith::CmpIPredicate::ugt>;
    using CmpIntGeOp =
        CmpIntInequalityOp<arith::CmpIPredicate::sge, arith::CmpIPredicate::uge>;
    RewritePatternSet patterns(&getContext());
    patterns.insert<
        CastOpConversion,                     //
        ConstantOpConversion,                 //
        FuncOpConversion<FuncOp>,             //
        FuncOpConversion<stdx::ClosureOp>,    //
        IndexOpConversion,                    //
        PragmaOpConversion,                   //
        ReshapeOpConversion,                  //
        ReturnOpConversion,                   //
        SpecialOpConversion<tile::ArgSortOp>, //
        SpecialOpConversion<tile::GatherOp>,  //
        SpecialOpConversion<tile::PrngOp>,    //
        SpecialOpConversion<tile::ScatterOp>, //
        ShapeOpConversion,                    //
        TraceOpConversion,                    //
        ScfForOpConversion,                   //
        ContractionOpConversion<CombinationKind::none, FirstOperand>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::arith::AddFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::arith::AddIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::arith::MulFOp>,
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::arith::MulIOp>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpFloatOp<arith::CmpFPredicate::OEQ>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpIntOp<arith::CmpIPredicate::eq>,
                                ComparandsAre<EltwiseInteger>>,
        ContractionOpConversion<
            CombinationKind::cond,
            ContractionCondOp<CmpFloatOp<arith::CmpFPredicate::OEQ>>,
            AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond,
                                ContractionCondOp<CmpIntOp<arith::CmpIPredicate::eq>>,
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
        EltwiseOpConversion<tile::CeilOp, StdOp<mlir::math::CeilOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::FloorOp, StdOp<stdx::FloorOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::RoundOp, StdOp<stdx::RoundOp>,
                            OperandsAre<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, StdOp<mlir::arith::NegFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::NegOp, NegIOp, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::arith::AddFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::AddOp, StdOp<mlir::arith::AddIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::arith::SubFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::SubOp, StdOp<mlir::arith::SubIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::arith::MulFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::MulOp, StdOp<mlir::arith::MulIOp>,
                            ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::arith::DivFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::arith::DivSIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::DivOp, StdOp<mlir::arith::DivUIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::SqrtOp, StdOp<math::SqrtOp>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::arith::RemFOp>,
                            ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::arith::RemSIOp>,
                            ResultIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::ModOp, StdOp<mlir::arith::RemUIOp>,
                            ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpFloatOp<arith::CmpFPredicate::OEQ>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpEqOp, CmpIntOp<arith::CmpIPredicate::eq>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpFloatOp<arith::CmpFPredicate::ONE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpNeOp, CmpIntOp<arith::CmpIPredicate::ne>,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpFloatOp<arith::CmpFPredicate::OLT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLtOp, CmpIntLtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpFloatOp<arith::CmpFPredicate::OLE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpLeOp, CmpIntLeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpFloatOp<arith::CmpFPredicate::OGT>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGtOp, CmpIntGtOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpFloatOp<arith::CmpFPredicate::OGE>,
                            AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<tile::CmpGeOp, CmpIntGeOp,
                            OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<tile::BitAndOp, StdOp<mlir::arith::AndIOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitOrOp, StdOp<mlir::arith::OrIOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitNotOp, NotOp>,
        EltwiseOpConversion<tile::BitXorOp, StdOp<mlir::arith::XOrIOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShlOp, StdOp<mlir::arith::ShLIOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::arith::ShRSIOp>,
                            FirstOperandIs<EltwiseSigned>>,
        EltwiseOpConversion<tile::BitShrOp, StdOp<mlir::arith::ShRUIOp>,
                            FirstOperandIs<EltwiseUnsigned>>,
        EltwiseOpConversion<tile::LogicalAndOp, LogicalOp<mlir::arith::AndIOp>>,
        EltwiseOpConversion<tile::LogicalNotOp, LogicalNotOp>,
        EltwiseOpConversion<tile::LogicalOrOp, LogicalOp<mlir::arith::OrIOp>>,
        EltwiseOpConversion<tile::LogicalXorOp, LogicalOp<mlir::arith::XOrIOp>>,
        EltwiseOpConversion<tile::ReluOp, StdOp<stdx::ReluOp>>,
        EltwiseOpConversion<tile::SelectOp, SelectOp>,
        EltwiseOpConversion<tile::IdentOp, FirstOperand>>(&getContext());

    // Run the conversion
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
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
