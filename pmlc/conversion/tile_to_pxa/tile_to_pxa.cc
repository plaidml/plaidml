// Copyright 2020, Intel Corporation

#include "pmlc/conversion/tile_to_pxa/tile_to_pxa.h"

#include <utility>

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/eltwise/ir/dialect.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/ir/dialect.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::tile_to_pxa {

namespace ew = dialect::eltwise;
namespace pxa = dialect::pxa;

using dialect::eltwise::ScalarType;
using dialect::tile::AffineConstantOp;
using dialect::tile::AggregationKind;
using dialect::tile::CombinationKind;
using dialect::tile::ContractionOp;
using dialect::tile::ContractionOpOperandAdaptor;
using dialect::tile::IndexOp;
using dialect::tile::ShapeOp;
using dialect::tile::ShapeOpOperandAdaptor;
using dialect::tile::TraceOp;
using util::DataType;

using llvm::Optional;
using llvm::SmallVector;
using mlir::AffineConstantExpr;
using mlir::AffineIfOp;
using mlir::AffineLoadOp;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineStoreOp;
using mlir::AllocOp;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::CallOp;
using mlir::CmpFPredicate;
using mlir::CmpIPredicate;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::FlatSymbolRefAttr;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::OwningRewritePatternList;
using mlir::Pattern;
using mlir::PatternMatchResult;
using mlir::RankedTensorType;
using mlir::ReturnOp;
using mlir::StringAttr;
using mlir::SymbolRefAttr;
using mlir::Type;
using mlir::Value;

namespace {

struct TypeConverter : public mlir::TypeConverter {
  using mlir::TypeConverter::convertType;

  Type convertType(Type type) final {
    IVLOG(2, "TypeConverter::convertType> " << mlir::debugString(type));
    if (type.isa<FunctionType>()) {
      IVLOG(4, "  FunctionType");
      return type;
    }
    if (auto scalarType = type.dyn_cast<ScalarType>()) {
      return scalarType.toStandard();
    }
    if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
      IVLOG(4, "  RankedTensorType");
      return MemRefType::get(rankedTensorType.getShape(),
                             convertType(rankedTensorType.getElementType()));
    }
    return {};
  }
};

ScalarType getScalarType(Type type) {
  if (auto tensorType = type.dyn_cast<mlir::TensorType>()) {
    type = tensorType.getElementType();
  }
  return type.cast<ScalarType>();
}

ScalarType getScalarType(Value value) { return getScalarType(value.getType()); }

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();
    IVLOG(2, "FuncOpConversion::rewrite> " << mlir::debugString(type));

    // Convert the function signature
    TypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() +
                                                    type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    for (unsigned i = 0; i < type.getNumResults(); ++i) {
      result.addInputs({typeConverter.convertType(type.getResult(i))});
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(result.getConvertedTypes(), llvm::None,
                                    op.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct AffineConstantOpConversion
    : public OpConversionPattern<AffineConstantOp> {
  using OpConversionPattern<AffineConstantOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(AffineConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto value = op.getValue().cast<IntegerAttr>().getInt();
    rewriter.replaceOpWithNewOp<mlir::ConstantIndexOp>(op, value);
    return matchSuccess();
  }
};

struct ScalarConstantOpConversion
    : public OpConversionPattern<ew::ScalarConstantOp> {
  using OpConversionPattern<ew::ScalarConstantOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(ew::ScalarConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto stdType = getScalarType(op).toStandard();
    auto value = op.getValue();
    if (auto floatType = stdType.dyn_cast<FloatType>()) {
      auto floatAttr = value.cast<FloatAttr>();
      value = FloatAttr::get(floatType, floatAttr.getValueAsDouble());
    } else if (auto intType = stdType.dyn_cast<IntegerType>()) {
      auto intAttr = value.cast<IntegerAttr>();
      value = IntegerAttr::get(intType, intAttr.getInt());
    } else {
      llvm_unreachable("Invalid scalar constant op");
    }
    rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, stdType, value);
    return matchSuccess();
  }
};

Value createCastOp(ConversionPatternRewriter &rewriter, Location loc,
                   Value from, Type intoType, bool isSigned) {
  auto fromType = from.getType();
  if (fromType == intoType) {
    return from;
  }
  if (auto intoFloatType = intoType.dyn_cast<FloatType>()) {
    if (auto fromFloatType = fromType.dyn_cast<FloatType>()) {
      if (fromFloatType.getWidth() < intoFloatType.getWidth()) {
        // FPExtOp: FloatType -> wider FloatType
        return rewriter.create<mlir::FPExtOp>(loc, from, intoType).getResult();
      }
      // FPTruncOp: FloatType -> narrower FloatType
      return rewriter.create<mlir::FPTruncOp>(loc, from, intoType).getResult();
    }
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      // SIToFPOp: IntegerType -> FloatType
      return rewriter.create<mlir::SIToFPOp>(loc, from, intoType).getResult();
    }
  }
  if (auto intoIntType = intoType.dyn_cast<IntegerType>()) {
    if (auto fromIntType = fromType.dyn_cast<IntegerType>()) {
      if (fromIntType.getWidth() < intoIntType.getWidth()) {
        if (isSigned) {
          // SignExtendIOp: IntegerType -> wider signed int
          return rewriter.create<mlir::SignExtendIOp>(loc, from, intoType)
              .getResult();
        }
        // ZeroExtendIOp: IntegerType -> wider unsigned int
        return rewriter.create<mlir::ZeroExtendIOp>(loc, from, intoType)
            .getResult();
      }
      // TruncateIOp: IntegerType -> narrower IntegerType
      return rewriter.create<mlir::TruncateIOp>(loc, from, intoType)
          .getResult();
    }
  }
  llvm_unreachable("Unsupported cast op");
}

struct Matcher {
  static PatternMatchResult
  matchSuccess(std::unique_ptr<mlir::PatternState> state = {}) {
    return PatternMatchResult(std::move(state));
  }

  PatternMatchResult operator()(Operation *op) {
    return match(op) ? matchSuccess() : llvm::None;
  }

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
    ContractionOpOperandAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) ||
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
  bool match(Type type) const { return isFloat(getScalarType(type).type()); }
};

struct EltwiseInteger {
  bool match(Type type) const { return isInteger(getScalarType(type).type()); }
};

struct EltwiseSigned {
  bool match(Type type) const { return isSigned(getScalarType(type).type()); }
};

struct EltwiseUnsigned {
  bool match(Type type) const { return isUnsigned(getScalarType(type).type()); }
};

struct FirstOperand {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    return operands.front();
  }
};

DataType promoteTypes(ConversionPatternRewriter &rewriter, Location loc,
                      ArrayRef<Value> operands, ArrayRef<DataType> types,
                      llvm::SmallVectorImpl<Value> *into) {
  // First, determine the 'final' type that wins the promotion
  DataType bestType = DataType::invalid;
  for (auto type : types) {
    bestType = promoteTypes(bestType, type);
  }
  // Next, cast each operand to the 'final' type
  auto scalarType = rewriter.getType<ScalarType>(bestType);
  auto targetType = scalarType.toStandard();
  for (unsigned i = 0; i < operands.size(); i++) {
    auto dtype = types[i];
    auto operand = operands[i];
    auto castOp =
        createCastOp(rewriter, loc, operand, targetType, isSigned(dtype));
    into->push_back(castOp);
  }
  return bestType;
}

template <typename OpType>
struct StdOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
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
               ArrayRef<DataType> types) {
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
               ArrayRef<DataType> types) {
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
               ArrayRef<DataType> types) {
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
               ArrayRef<DataType> types) {
    SmallVector<Value, 2> promoted;
    auto dataType = promoteTypes(rewriter, loc, operands, types, &promoted);
    auto predicate = isSigned(dataType) ? signedPred : unsignedPred;
    return rewriter
        .create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <typename CmpOpBuilder>
struct CondOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(rewriter, loc, resultType,
                                   operands.take_front(2), types.take_front(2));
    auto zero = createInit(rewriter, loc, resultType, AggregationKind::add);
    return rewriter.create<mlir::SelectOp>(loc, cmp, operands[2], zero)
        .getResult();
  }

  Value createInit(OpBuilder &builder, Location loc, Type type,
                   AggregationKind agg) const {
    if (auto floatType = type.dyn_cast<FloatType>()) {
      switch (agg) {
      case AggregationKind::add:
        return builder.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(0.0),
                                                     floatType);
      case AggregationKind::mul:
        return builder.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(1.0),
                                                     floatType);
      default:
        llvm_unreachable("Unsupported aggregation for createInit");
      }
    } else if (auto intType = type.dyn_cast<IntegerType>()) {
      switch (agg) {
      case AggregationKind::add:
        return builder.create<mlir::ConstantIntOp>(loc, 0, intType);
      case AggregationKind::mul:
        return builder.create<mlir::ConstantIntOp>(loc, 1, intType);
      default:
        llvm_unreachable("Unsupported aggregation for createInit");
      }
    }
    llvm_unreachable("Unknown type for createInit");
  }
};

static Value buildBroadcastLoad(OpBuilder &builder, Location loc, Value operand,
                                size_t outRank) {
  auto body = builder.getBlock();
  auto defOp = operand.getDefiningOp();
  Attribute attr;
  // Handle scalar values
  if (defOp && mlir::m_Constant(&attr).match(defOp)) {
    return operand;
  }
  // handle broadcasts
  auto operandType = operand.getType().cast<MemRefType>();
  assert(operandType.getRank() <= outRank && "result rank < operand rank");
  auto op_shape = operandType.getShape();
  SmallVector<Value, 8> operandIdxs(operandType.getRank());
  for (unsigned i = 0; i < operandType.getRank(); i++) {
    unsigned j = outRank - i - 1;
    unsigned k = operandType.getRank() - i - 1;
    if (op_shape[k] == 1) {
      operandIdxs[k] = builder.create<mlir::ConstantIndexOp>(loc, 0);
    } else {
      operandIdxs[k] = body->getArgument(j);
    }
  }
  return builder.create<AffineLoadOp>(loc, operand, operandIdxs);
}

static void buildSimpleStore(OpBuilder &builder, Location loc, Value scalar,
                             Value memRef) {
  auto body = builder.getBlock();
  // TODO: Maybe fix ValueRange to support arguments?
  SmallVector<Value, 8> idxs;
  for (size_t i = 0; i < body->getNumArguments(); i++) {
    idxs.push_back(body->getArgument(i));
  }
  builder.create<AffineStoreOp>(loc, scalar, memRef, idxs);
}

template <typename FromOpType, typename IntoOpBuilder,
          typename Matcher = AlwaysTrue>
struct EltwiseOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  PatternMatchResult match(Operation *op) const final {
    IVLOG(2, "EltwiseOpConversion::match>");
    Matcher pred;
    return pred(op);
  }

  void rewrite(FromOpType op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const final {
    TypeConverter typeConverter;
    auto loc = op.getLoc();
    auto resultType = op.result().getType();
    auto resultMemRefType =
        typeConverter.convertType(resultType).template cast<MemRefType>();

    // Allocate the result
    auto resultMemRef =
        rewriter.create<AllocOp>(loc, resultMemRefType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<pxa::AffineParallelOp>(
        loc, resultMemRefType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the loads
    SmallVector<Value, 4> scalars;
    for (size_t i = 0; i < operands.size(); i++) {
      scalars.push_back(buildBroadcastLoad(rewriter, loc, operands[i],
                                           resultMemRefType.getRank()));
    }

    // Create the standard op
    auto elementType = resultMemRefType.getElementType();
    SmallVector<DataType, 4> operandDataTypes;
    for (auto type : op.getOperation()->getOperandTypes()) {
      auto scalarType = getScalarType(type);
      operandDataTypes.push_back(scalarType.type());
    }
    IntoOpBuilder intoOpBuilder;
    auto result = intoOpBuilder.create(rewriter, loc, elementType, scalars,
                                       operandDataTypes);

    // Create the store
    buildSimpleStore(rewriter, loc, result, resultMemRef);

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, resultMemRef);
  }
};

template <CombinationKind comboKind, typename ComboBuilder,
          typename Matcher = AlwaysTrue>
struct ContractionOpConversion : public OpConversionPattern<ContractionOp> {
  using OpConversionPattern<ContractionOp>::OpConversionPattern;

  PatternMatchResult match(Operation *op) const final {
    IVLOG(2, "ContractionOpConversion::match>");
    if (auto cionOp = llvm::dyn_cast<ContractionOp>(op)) {
      if (cionOp.combo() != comboKind) {
        return matchFailure();
      }
      if (!cionOp.lower_bounds().hasValue() ||
          !cionOp.upper_bounds().hasValue()) {
        cionOp.emitError("contraction bounds must be computed");
        return matchFailure();
      }
      Matcher pred;
      return pred(cionOp);
    }
    return matchFailure();
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
    ContractionOpOperandAdaptor cionAdaptor(operands);
    auto cionOperands = cionAdaptor.operands();

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Determine ranges
    SmallVector<int64_t, 8> ranges;
    auto lowerBounds = op.lower_bounds().getValue();
    auto upperBounds = op.upper_bounds().getValue();
    assert(lowerBounds.getNumResults() == upperBounds.getNumResults() &&
           "mismatched dims for lower and upper bounds");
    for (unsigned i = 0; i < lowerBounds.getNumResults(); i++) {
      auto rangeExpr = upperBounds.getResult(i) - lowerBounds.getResult(i) + 1;
      auto range = rangeExpr.cast<AffineConstantExpr>().getValue();
      ranges.emplace_back(range);
    }

    // Do initialization
    auto initFor =
        rewriter.create<pxa::AffineParallelOp>(loc, resultType.getShape());
    auto initForBuilder = initFor.getBodyBuilder();
    auto initLoad = buildBroadcastLoad(initForBuilder, loc, cionAdaptor.init(),
                                       resultType.getRank());
    buildSimpleStore(initForBuilder, loc, initLoad, resultMemRef);

    // Make the outer loops
    auto forOp = rewriter.create<pxa::AffineParallelOp>(loc, ranges);
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    // TODO: Maybe fix ValueRange?
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < body->getNumArguments(); i++) {
      idxs.push_back(body->getArgument(i));
    }

    // add constraints
    if (op.cons()) {
      auto cons = op.cons().getValue();
      auto ifOp = rewriter.create<AffineIfOp>(loc, cons, idxs, false);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
    }

    // Create the loads + casts
    SmallVector<Value, 4> scalars;
    auto srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      auto operand = cionOperands[i];
      auto defOp = operand.getDefiningOp();
      Attribute attr;
      if (defOp && mlir::m_Constant(&attr).match(defOp)) {
        scalars.push_back(operand);
      } else {
        auto map = srcs[i].cast<AffineMapAttr>().getValue();
        scalars.push_back(
            rewriter.create<AffineLoadOp>(loc, operand, map, idxs));
      }
    }

    // Do the combination op
    ComboBuilder comboBuilder;
    auto elementType = resultType.getElementType();
    SmallVector<DataType, 4> operandDataTypes;
    for (auto type : op.operands().getTypes()) {
      auto scalarType = getScalarType(type);
      operandDataTypes.push_back(scalarType.type());
    }
    auto combined = comboBuilder.create(rewriter, loc, elementType, scalars,
                                        operandDataTypes);

    // Create the store
    auto resultMap = op.sink();
    if (resultMap.isEmpty()) {
      SmallVector<Value, 0> emptyIdxs;
      rewriter.create<pxa::AffineReduceOp>(loc, op.agg(), combined,
                                           resultMemRef, resultMap, emptyIdxs);
    } else {
      rewriter.create<pxa::AffineReduceOp>(loc, op.agg(), combined,
                                           resultMemRef, resultMap, idxs);
    }

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);
  }
};

struct IndexOpConversion : public OpConversionPattern<IndexOp> {
  using OpConversionPattern<IndexOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(IndexOp op, llvm::ArrayRef<Value> operands,
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
    auto forOp =
        rewriter.create<pxa::AffineParallelOp>(loc, resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    // TODO: Maybe fix ValueRange?
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < body->getNumArguments(); i++) {
      idxs.push_back(body->getArgument(i));
    }

    // Load the index value
    // TODO: add check that dim is within range in verifier
    auto dim = op.dim().getZExtValue();
    auto map = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)});
    auto apply = rewriter.create<mlir::AffineApplyOp>(loc, map, idxs[dim]);

    // Create the store
    auto cast = rewriter.create<mlir::IndexCastOp>(loc, apply,
                                                   rewriter.getIntegerType(32));
    rewriter.create<AffineStoreOp>(loc, cast, resultMemRef, idxs);

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);

    return matchSuccess();
  }
};

struct ShapeOpConversion : public OpConversionPattern<ShapeOp> {
  using OpConversionPattern<ShapeOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(ShapeOp op, llvm::ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "ShapeOpConversion::matchAndRewrite>");

    // Create an adaptor
    ShapeOpOperandAdaptor adaptor(operands);

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Populate the buffer with the shape dims
    auto operandType = adaptor.tensor().getType().cast<MemRefType>();
    for (unsigned i = 0; i < operandType.getRank(); i++) {
      auto idx = rewriter.create<mlir::ConstantIndexOp>(loc, i);
      auto dim = rewriter.create<mlir::DimOp>(loc, adaptor.tensor(), i);
      auto cast = rewriter.create<mlir::IndexCastOp>(
          loc, dim, rewriter.getIntegerType(32));
      SmallVector<Value, 1> idxs = {idx};
      rewriter.create<mlir::StoreOp>(loc, cast, resultMemRef, idxs);
    }

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);

    return matchSuccess();
  }
};

struct CastOpConversion : public OpConversionPattern<ew::CastOp> {
  using OpConversionPattern<ew::CastOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(ew::CastOp op, llvm::ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IVLOG(2, "CastOpConversion::matchAndRewrite>");

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();
    auto operand = operands[0];
    auto operandType = operand.getType().cast<MemRefType>();
    if (resultType == operandType) {
      rewriter.replaceOp(op, operand);
      return matchSuccess();
    }

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp =
        rewriter.create<pxa::AffineParallelOp>(loc, resultType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    // TODO: Maybe fix ValueRange?
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < body->getNumArguments(); i++) {
      idxs.push_back(body->getArgument(i));
    }

    // Create the load
    auto scalar = rewriter.create<AffineLoadOp>(loc, operand, idxs);

    // Create the standard cast op
    auto scalarType = getScalarType(op.tensor());
    auto dtype = scalarType.type();
    auto result = createCastOp(rewriter, loc, scalar,
                               resultType.getElementType(), isSigned(dtype));

    // Create the store
    rewriter.create<AffineStoreOp>(loc, result, resultMemRef, idxs);

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);

    return matchSuccess();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    IVLOG(2, "ReturnOpConversion::matchAndRewrite>");
    auto &block = op.getParentRegion()->front();
    auto funcOp = op.getParentOfType<FuncOp>();
    auto blockArg = funcOp.getType().getNumInputs() - op.getNumOperands();
    for (auto operand : operands) {
      operand.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return matchSuccess();
  }
};

struct TraceOpConversion : public OpConversionPattern<TraceOp> {
  using OpConversionPattern<TraceOp>::OpConversionPattern;

  PatternMatchResult
  matchAndRewrite(TraceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto module = op.getParentOfType<ModuleOp>();
    auto symbol = createStubFunc(module, op.msgAttr());
    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{});
    rewriter.replaceOp(op, op.tensor());
    return matchSuccess();
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

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() final {
    // Set up target (i.e. what is legal)
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::AffineOpsDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<dialect::pxa::Dialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      auto funcType = op.getType();
      return funcType.getNumResults() == 0;
    });
    target.addDynamicallyLegalOp<ReturnOp>(
        [](ReturnOp op) { return op.getNumOperands() == 0; });

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
        AffineConstantOpConversion, CastOpConversion, FuncOpConversion,
        IndexOpConversion, ReturnOpConversion, ScalarConstantOpConversion,
        ShapeOpConversion,
        TraceOpConversion, // TODO: PrngOpConversion
                           // TODO: SpecialOpConversion (GatherOp, ReshapeOp,
                           // ScatterOp, ZeroOp)
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
                                ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::eq,
                                CmpIntOp<CmpIPredicate::eq>,
                                ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpFloatOp<CmpFPredicate::OEQ>>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpIntOp<CmpIPredicate::eq>>,
                                AnyComparandIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::ExpOp, StdOp<mlir::ExpOp>>,
        EltwiseOpConversion<ew::NegOp, StdOp<mlir::NegFOp>,
                            ResultIs<EltwiseFloat>>,
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
        EltwiseOpConversion<ew::BitXorOp, StdOp<mlir::XOrOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitShlOp, StdOp<mlir::ShiftLeftOp>,
                            OperandsAre<EltwiseInteger>>,
        EltwiseOpConversion<ew::BitShrOp, StdOp<mlir::SignedShiftRightOp>,
                            FirstOperandIs<EltwiseSigned>>,
        EltwiseOpConversion<ew::BitShrOp, StdOp<mlir::UnsignedShiftRightOp>,
                            FirstOperandIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::SelectOp, SelectOp>,
        EltwiseOpConversion<ew::IdentOp, FirstOperand>>(&getContext());

    // Run the conversion
    if (failed(applyFullConversion(getModule(), target, patterns, nullptr))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerTileToPXAPass() {
  return std::make_unique<LoweringPass>();
}

static mlir::PassRegistration<LoweringPass>
    legalize_pass("convert-tile-to-pxa", "Convert Tile dialect to PXA dialect");

} // namespace pmlc::conversion::tile_to_pxa
