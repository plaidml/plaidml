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

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/ir/dialect.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/ir/dialect.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/contraction.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::tile_to_pxa {

namespace ew = dialect::eltwise;
namespace pxa = dialect::pxa;

using dialect::eltwise::ScalarType;
using dialect::tile::AffineConstantOp;
using dialect::tile::AggregationKind;
using dialect::tile::CombinationKind;
using dialect::tile::Contraction;
using dialect::tile::ContractionOp;
using dialect::tile::ContractionOpOperandAdaptor;
using dialect::tile::IndexOp;
using dialect::tile::Shape;
using dialect::tile::ShapeOp;
using dialect::tile::ShapeOpOperandAdaptor;
using ::vertexai::tile::DataType;

using llvm::Optional;
using llvm::SmallVector;
using mlir::AffineLoadOp;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::AffineStoreOp;
using mlir::AffineTerminatorOp;
using mlir::AllocOp;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::CmpFPredicate;
using mlir::CmpIPredicate;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::Location;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::OwningRewritePatternList;
using mlir::Pattern;
using mlir::PatternMatchResult;
using mlir::RankedTensorType;
using mlir::ReturnOp;
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
      return MemRefType::get(rankedTensorType.getShape(), convertType(rankedTensorType.getElementType()));
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

ScalarType getScalarType(Value value) { return getScalarType(value->getType()); }

Shape getShape(Type type) {
  auto rankedTensorType = ew::getRankedTensorType(type);
  return rankedTensorType.getShape();
}

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      FuncOp op,                       //
      ArrayRef<Value> operands,        //
      ConversionPatternRewriter& rewriter) const final {
    FunctionType type = op.getType();
    IVLOG(2, "FuncOpConversion::rewrite> " << mlir::debugString(type));

    // Convert the function signature
    TypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() + type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    for (unsigned i = 0; i < type.getNumResults(); ++i) {
      result.addInputs({typeConverter.convertType(type.getResult(i))});
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(result.getConvertedTypes(), llvm::None, op.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);

    return matchSuccess();
  }
};

struct AffineConstantOpConversion : public OpConversionPattern<AffineConstantOp> {
  using OpConversionPattern<AffineConstantOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      AffineConstantOp op,             //
      ArrayRef<Value> operands,        //
      ConversionPatternRewriter& rewriter) const final {
    auto value = op.getValue().cast<IntegerAttr>().getInt();
    auto newOp = rewriter.create<mlir::ConstantIndexOp>(op.getLoc(), value);
    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

struct ScalarConstantOpConversion : public OpConversionPattern<ew::ScalarConstantOp> {
  using OpConversionPattern<ew::ScalarConstantOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      ew::ScalarConstantOp op,         //
      ArrayRef<Value> operands,        //
      ConversionPatternRewriter& rewriter) const final {
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
    auto newOp = rewriter.create<mlir::ConstantOp>(op.getLoc(), stdType, value);
    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

Value createCastOp(ConversionPatternRewriter& rewriter, Location loc, Value from, Type intoType, bool isSigned) {
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
          return rewriter.create<mlir::SignExtendIOp>(loc, from, intoType).getResult();
        }
        // ZeroExtendIOp: IntegerType -> wider unsigned int
        return rewriter.create<mlir::ZeroExtendIOp>(loc, from, intoType).getResult();
      }
      // TruncateIOp: IntegerType -> narrower IntegerType
      return rewriter.create<mlir::TruncateIOp>(loc, from, intoType).getResult();
    }
  }
  llvm_unreachable("Unsupported cast op");
}

struct Matcher {
  static PatternMatchResult matchSuccess(std::unique_ptr<mlir::PatternState> state = {}) {
    return PatternMatchResult(std::move(state));
  }

  PatternMatchResult operator()(Operation* op) { return match(op) ? matchSuccess() : llvm::None; }

  virtual bool match(Operation* op) const { return false; }
};

struct AlwaysTrue : Matcher {
  bool match(Operation* op) const final { return true; }
};

template <typename InnerPredicate>
struct ResultIs : Matcher {
  bool match(Operation* op) const final {
    InnerPredicate pred;
    return pred.match(op->getResult(0).getType());
  }
};

template <typename InnerPredicate>
struct AnyOperandIs : Matcher {
  bool match(Operation* op) const final {
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
  bool match(Operation* op) const final {
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
struct AnyComparandIs : Matcher {
  bool match(Operation* op) const final {
    SmallVector<Value, 4> allOperands(op->getOperands());
    ContractionOpOperandAdaptor adaptor(allOperands);
    auto operands = adaptor.operands();
    InnerPredicate pred;
    return pred.match(operands[0].getType()) || pred.match(operands[1].getType());
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
  bool match(Type type) const { return is_float(getScalarType(type).type()); }
};

struct EltwiseInteger {
  bool match(Type type) const { return is_int(getScalarType(type).type()); }
};

struct EltwiseUnsigned {
  bool match(Type type) const { return is_uint(getScalarType(type).type()); }
};

struct FirstOperand {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    return operands.front();
  }
};

// Returns the Plaid arithmetic conversion rank of a type.
unsigned getDataTypeRank(DataType dtype) {
  switch (dtype) {
    case DataType::INVALID:
      return 0;
    case DataType::BOOLEAN:
      return 2;
    case DataType::INT8:
      return 3;
    case DataType::UINT8:
      return 4;
    case DataType::INT16:
      return 5;
    case DataType::UINT16:
      return 6;
    case DataType::INT32:
      return 7;
    case DataType::UINT32:
      return 8;
    case DataType::INT64:
      return 9;
    case DataType::UINT64:
      return 10;
    case DataType::FLOAT16:
      return 11;
    case DataType::FLOAT32:
      return 12;
    case DataType::FLOAT64:
      return 13;
    default:
      throw std::logic_error{"Invalid type found in typecheck"};
  }
}

DataType promoteTypes(ConversionPatternRewriter& rewriter, Location loc, ArrayRef<Value> operands,
                      ArrayRef<DataType> types, llvm::SmallVectorImpl<Value>* into) {
  // First, determine the 'final' type that wins the promotion
  DataType bestType = DataType::INVALID;
  for (auto type : types) {
    IVLOG(1, "type: " << to_string(type));
    if (getDataTypeRank(type) > getDataTypeRank(bestType)) {
      IVLOG(1, "new best");
      bestType = type;
    }
  }
  // Next, cast each operand to the 'final' type
  auto isSigned = is_int(bestType);
  auto scalarType = rewriter.getType<ScalarType>(bestType);
  auto targetType = scalarType.toStandard();
  for (auto operand : operands) {
    auto castOp = createCastOp(rewriter, loc, operand, targetType, isSigned);
    into->push_back(castOp);
  }
  return bestType;
}

template <typename OpType>
struct StdOp {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
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
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands.drop_front(), types.drop_front(), &promoted);
    auto op = rewriter.create<mlir::SelectOp>(loc, operands[0], promoted[0], promoted[1]);
    return op.getResult();
  }
};

template <CmpFPredicate predicate>
struct CmpFloatOp {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter.create<mlir::CmpFOp>(loc, predicate, promoted[0], promoted[1]).getResult();
  }
};

template <CmpIPredicate predicate>
struct CmpIntOp {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter.create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1]).getResult();
  }
};

template <CmpIPredicate signedPred, CmpIPredicate unsignedPred>
struct CmpIntInequalityOp {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    IVLOG(1, "CmpIntInequalityOp");
    SmallVector<Value, 2> promoted;
    auto dataType = promoteTypes(rewriter, loc, operands, types, &promoted);
    IVLOG(1, "dataType: " << to_string(dataType));
    auto predicate = is_int(dataType) ? signedPred : unsignedPred;
    return rewriter.create<mlir::CmpIOp>(loc, predicate, promoted[0], promoted[1]).getResult();
  }
};

template <typename CmpOpBuilder>
struct CondOp {
  Value create(ConversionPatternRewriter& rewriter, Location loc, Type resultType, ArrayRef<Value> operands,
               ArrayRef<DataType> types) {
    CmpOpBuilder cmpOpBuilder;
    auto cmp = cmpOpBuilder.create(rewriter, loc, resultType, operands.take_front(2), types.take_front(2));
    auto zero = createInit(rewriter, loc, resultType, AggregationKind::add);
    return rewriter.create<mlir::SelectOp>(loc, cmp, operands[2], zero).getResult();
  }

  Value createInit(OpBuilder& builder, Location loc, Type type, AggregationKind agg) const {
    if (auto floatType = type.dyn_cast<FloatType>()) {
      switch (agg) {
        case AggregationKind::add:
          return builder.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(0.0), floatType);
        case AggregationKind::mul:
          return builder.create<mlir::ConstantFloatOp>(loc, llvm::APFloat(1.0), floatType);
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

template <typename FromOpType, typename IntoOpBuilder, typename Matcher = AlwaysTrue>
struct EltwiseOpConversion : public OpConversionPattern<FromOpType> {
  using OpConversionPattern<FromOpType>::OpConversionPattern;

  PatternMatchResult match(Operation* op) const final {
    Matcher pred;
    return pred(op);
  }

  void rewrite(                  //
      FromOpType op,             //
      ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const final {
    TypeConverter typeConverter;
    auto loc = op.getLoc();
    auto resultType = op.result()->getType();
    auto resultMemRefType = typeConverter.convertType(resultType).template cast<MemRefType>();

    // Allocate the result
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultMemRefType).getResult();

    // Make a parallel for loop to fill the result
    auto ranges = rewriter.getI64ArrayAttr(resultMemRefType.getShape());
    auto dynamicRanges = ArrayRef<Value>();
    auto forOp = rewriter.create<pxa::AffineParallelForOp>(loc, ranges, dynamicRanges);
    auto body = rewriter.createBlock(&forOp.inner());
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < ranges.size(); i++) {
      auto idx = body->addArgument(rewriter.getIndexType());
      idxs.push_back(idx);
    }

    // Create the loads
    SmallVector<Value, 4> scalars;
    for (size_t i = 0; i < operands.size(); i++) {
      auto operand = operands[i];
      auto defOp = operand.getDefiningOp();
      Attribute attr;
      if (defOp && mlir::m_Constant(&attr).match(defOp)) {
        scalars.push_back(operand);
      } else {
        // handle broadcasts
        auto operandType = operand.getType().cast<MemRefType>();
        assert(operandType.getRank() <= resultMemRefType.getRank() && "result rank < operand rank");
        SmallVector<Value, 8> operandIdxs(operandType.getRank());
        for (unsigned i = 0; i < operandType.getRank(); i++) {
          unsigned j = resultMemRefType.getRank() - i - 1;
          unsigned k = operandType.getRank() - i - 1;
          operandIdxs[k] = body->getArgument(j);
        }
        scalars.push_back(rewriter.create<AffineLoadOp>(loc, operand, operandIdxs));
      }
    }

    // Create the standard op
    auto elementType = resultMemRefType.getElementType();
    SmallVector<DataType, 4> operandDataTypes;
    for (auto type : op.getOperation()->getOperandTypes()) {
      auto scalarType = getScalarType(type);
      operandDataTypes.push_back(scalarType.type());
    }
    IntoOpBuilder intoOpBuilder;
    auto result = intoOpBuilder.create(rewriter, loc, elementType, scalars, operandDataTypes);

    // Create the store
    rewriter.create<AffineStoreOp>(loc, result, resultMemRef, idxs);

    // Terminate the inner body
    rewriter.create<AffineTerminatorOp>(loc);

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, resultMemRef);
  }
};

template <CombinationKind comboKind, typename ComboBuilder, typename Matcher = AlwaysTrue>
struct ContractionOpConversion : public OpConversionPattern<ContractionOp> {
  using OpConversionPattern<ContractionOp>::OpConversionPattern;

  PatternMatchResult match(Operation* op) const final {
    if (auto cionOp = llvm::dyn_cast<ContractionOp>(op)) {
      if (cionOp.combo() != comboKind) {
        return matchFailure();
      }
      Matcher pred;
      return pred(cionOp);
    }
    return matchFailure();
  }

  void rewrite(                  //
      ContractionOp op,          //
      ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const final {
    try {
      tryRewrite(op, operands, rewriter);
    } catch (const std::exception& ex) {
      op.emitError(ex.what());
    }
  }

  void tryRewrite(               //
      ContractionOp op,          //
      ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const {
    // Create an adaptor
    ContractionOpOperandAdaptor cionAdaptor(operands);
    auto cionOperands = cionAdaptor.operands();

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Get the shape
    SmallVector<Shape, 4> shapes{getShape(op.result()->getType())};
    for (auto src : op.operands()) {
      shapes.emplace_back(getShape(src->getType()));
    }

    // Do the actual maths
    Contraction contraction{op};
    bool no_reduce = op.no_reduce().hasValue();
    const auto& [bounds, constraints] = contraction.ComputeBounds(shapes, no_reduce);

    // Extract ranges
    SmallVector<int64_t, 8> ranges;
    for (const auto& [key, value] : bounds) {
      uint64_t range = value.max - value.min + 1;
      ranges.emplace_back(range);
    }

    // TODO: addInitializer

    // Make the outer loops
    auto dynamicRanges = ArrayRef<Value>();
    auto forOp = rewriter.create<pxa::AffineParallelForOp>(loc, rewriter.getI64ArrayAttr(ranges), dynamicRanges);
    auto body = rewriter.createBlock(&forOp.inner());
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < ranges.size(); i++) {
      auto idx = body->addArgument(rewriter.getIndexType());
      idxs.push_back(idx);
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
        scalars.push_back(rewriter.create<AffineLoadOp>(loc, operand, map, idxs));
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
    auto combined = comboBuilder.create(rewriter, loc, elementType, scalars, operandDataTypes);

    // Create the store
    auto resultMap = op.sink();
    rewriter.create<pxa::AffineReduceOp>(loc, op.agg(), combined, resultMemRef, resultMap, idxs);

    // Terminate the inner body
    rewriter.create<AffineTerminatorOp>(loc);

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);
  }
};

struct IndexOpConversion : public OpConversionPattern<IndexOp> {
  using OpConversionPattern<IndexOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      IndexOp op,                      //
      llvm::ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "IndexOpConversion::matchAndRewrite>");

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto ranges = rewriter.getI64ArrayAttr(resultType.getShape());
    auto dynamicRanges = ArrayRef<Value>();
    auto forOp = rewriter.create<pxa::AffineParallelForOp>(loc, ranges, dynamicRanges);
    auto body = rewriter.createBlock(&forOp.inner());
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < ranges.size(); i++) {
      auto idx = body->addArgument(rewriter.getIndexType());
      idxs.push_back(idx);
    }

    // Load the index value
    // TODO: add check that dim is within range in verifier
    auto dim = op.dim().getZExtValue();
    auto map = AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)});
    auto apply = rewriter.create<mlir::AffineApplyOp>(loc, map, idxs[dim]);

    // Create the store
    auto cast = rewriter.create<mlir::IndexCastOp>(loc, apply, rewriter.getIntegerType(32));
    rewriter.create<AffineStoreOp>(loc, cast, resultMemRef, idxs);

    // Terminate the inner body
    rewriter.create<AffineTerminatorOp>(loc);

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);

    return matchSuccess();
  }
};

struct ShapeOpConversion : public OpConversionPattern<ShapeOp> {
  using OpConversionPattern<ShapeOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      ShapeOp op,                      //
      llvm::ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "ShapeOpConversion::matchAndRewrite>");

    // Create an adaptor
    ShapeOpOperandAdaptor adaptor(operands);

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Populate the buffer with the shape dims
    auto operandType = adaptor.tensor().getType().cast<MemRefType>();
    for (unsigned i = 0; i < operandType.getRank(); i++) {
      auto idx = rewriter.create<mlir::ConstantIndexOp>(loc, i);
      auto dim = rewriter.create<mlir::DimOp>(loc, adaptor.tensor(), i);
      auto cast = rewriter.create<mlir::IndexCastOp>(loc, dim, rewriter.getIntegerType(32));
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

  PatternMatchResult matchAndRewrite(  //
      ew::CastOp op,                   //
      llvm::ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "CastOpConversion::matchAndRewrite>");

    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();
    auto operand = operands[0];
    auto operandType = operand.getType().cast<MemRefType>();
    if (resultType == operandType) {
      rewriter.replaceOp(op, operand);
      return matchSuccess();
    }

    // Make an allocation for the output
    auto resultMemRef = rewriter.create<AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto ranges = rewriter.getI64ArrayAttr(resultType.getShape());
    auto dynamicRanges = ArrayRef<Value>();
    auto forOp = rewriter.create<pxa::AffineParallelForOp>(loc, ranges, dynamicRanges);
    auto body = rewriter.createBlock(&forOp.inner());
    SmallVector<Value, 8> idxs;
    for (size_t i = 0; i < ranges.size(); i++) {
      auto idx = body->addArgument(rewriter.getIndexType());
      idxs.push_back(idx);
    }

    // Create the load
    auto scalar = rewriter.create<AffineLoadOp>(loc, operand, idxs);

    // Create the standard cast op
    auto scalarType = getScalarType(op.tensor());
    bool isSigned = is_int(scalarType.type());
    auto result = createCastOp(rewriter, loc, scalar, resultType.getElementType(), isSigned);

    // Create the store
    rewriter.create<AffineStoreOp>(loc, result, resultMemRef, idxs);

    // Terminate the inner body
    rewriter.create<AffineTerminatorOp>(loc);

    // Replace the op
    rewriter.replaceOp(op, resultMemRef);

    return matchSuccess();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  PatternMatchResult matchAndRewrite(  //
      ReturnOp op,                     //
      ArrayRef<Value> operands,        //
      ConversionPatternRewriter& rewriter) const final {
    auto& block = op.getParentRegion()->front();
    auto funcOp = op.getParentOfType<FuncOp>();
    auto blockArg = funcOp.getType().getNumInputs() - op.getNumOperands();
    for (auto operand : operands) {
      operand.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return matchSuccess();
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
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto funcType = op.getType();
      return funcType.getNumResults() == 0;
    });
    target.addDynamicallyLegalOp<ReturnOp>([&](ReturnOp op) {  //
      return op.getNumOperands() == 0;
    });

    // Setup rewrite patterns
    using CmpIntLtOp = CmpIntInequalityOp<CmpIPredicate::slt, CmpIPredicate::ult>;
    using CmpIntLeOp = CmpIntInequalityOp<CmpIPredicate::sle, CmpIPredicate::ule>;
    using CmpIntGtOp = CmpIntInequalityOp<CmpIPredicate::sgt, CmpIPredicate::ugt>;
    using CmpIntGeOp = CmpIntInequalityOp<CmpIPredicate::sge, CmpIPredicate::uge>;
    OwningRewritePatternList patterns;
    patterns.insert<                 //
        AffineConstantOpConversion,  //
        CastOpConversion,            //
        FuncOpConversion,            //
        IndexOpConversion,           //
        ScalarConstantOpConversion,  //
        ShapeOpConversion,           //
        ReturnOpConversion,          //
        // TODO: PrngOpConversion
        // TODO: SpecialOpConversion (GatherOp, ReshapeOp, ScatterOp, ZeroOp)
        ContractionOpConversion<CombinationKind::none, FirstOperand>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddFOp>, ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::add, StdOp<mlir::AddIOp>, ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulFOp>, ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::mul, StdOp<mlir::MulIOp>, ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::eq, CmpFloatOp<CmpFPredicate::OEQ>, ResultIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::eq, CmpIntOp<CmpIPredicate::eq>, ResultIs<EltwiseInteger>>,
        ContractionOpConversion<CombinationKind::cond, CondOp<CmpFloatOp<CmpFPredicate::OEQ>>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond, CondOp<CmpIntOp<CmpIPredicate::eq>>,
                                AnyComparandIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::ExpOp, StdOp<mlir::ExpOp>>,
        EltwiseOpConversion<ew::NegOp, StdOp<mlir::NegFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::AddOp, StdOp<mlir::AddFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::AddOp, StdOp<mlir::AddIOp>, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::SubOp, StdOp<mlir::SubFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::SubOp, StdOp<mlir::SubIOp>, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::MulOp, StdOp<mlir::MulFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::MulOp, StdOp<mlir::MulIOp>, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::DivFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::SignedDivIOp>, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::DivOp, StdOp<mlir::UnsignedDivIOp>, ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::RemFOp>, ResultIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::SignedRemIOp>, ResultIs<EltwiseInteger>>,
        EltwiseOpConversion<ew::ModOp, StdOp<mlir::UnsignedRemIOp>, ResultIs<EltwiseUnsigned>>,
        EltwiseOpConversion<ew::CmpEqOp, CmpFloatOp<CmpFPredicate::OEQ>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpEqOp, CmpIntOp<CmpIPredicate::eq>, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpNeOp, CmpFloatOp<CmpFPredicate::ONE>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpNeOp, CmpIntOp<CmpIPredicate::ne>, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpLtOp, CmpFloatOp<CmpFPredicate::OLT>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpLtOp, CmpIntLtOp, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpLeOp, CmpFloatOp<CmpFPredicate::OLE>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpLeOp, CmpIntLeOp, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpGtOp, CmpFloatOp<CmpFPredicate::OGT>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpGtOp, CmpIntGtOp, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::CmpGeOp, CmpFloatOp<CmpFPredicate::OGE>, AnyOperandIs<EltwiseFloat>>,
        EltwiseOpConversion<ew::CmpGeOp, CmpIntGeOp, OperandsAre<Not<EltwiseFloat>>>,
        EltwiseOpConversion<ew::SelectOp, SelectOp>,    //
        EltwiseOpConversion<ew::IdentOp, FirstOperand>  //
        >(&getContext());

    // Run the conversion
    if (failed(applyFullConversion(getModule(), target, patterns, nullptr))) {
      getModule().dump();
      emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> pxa\n");
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createLowerTileToPXAPass() {  //
  return std::make_unique<LoweringPass>();
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "tile-legalize-to-pxa",                                 //
    "Legalize from Tile dialect to PXA dialect");

}  // namespace pmlc::conversion::tile_to_pxa
