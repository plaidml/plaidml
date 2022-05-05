// Copyright 2020, Intel Corporation

#include <limits>
#include <utility>

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

#include "pmlc/util/ident.h"

namespace pmlc::conversion::tile_to_pxa {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
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
    auto zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, resultType);
    auto neg = rewriter.create<mlir::arith::SubIOp>(loc, zero, operands[0]);
    return neg.getResult();
  }
};

struct NotOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    // -(x + 1) = -1 - x
    auto negOne = rewriter.create<mlir::arith::ConstantIntOp>(loc, -1, resultType);
    auto sub = rewriter.create<mlir::arith::SubIOp>(loc, negOne, operands[0]);
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

struct SelectOpBuilder {
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

template <arith::CmpFPredicate predicate>
struct CmpFloatOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::arith::CmpFOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <arith::CmpIPredicate predicate>
struct CmpIntOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    promoteTypes(rewriter, loc, operands, types, &promoted);
    return rewriter
        .create<mlir::arith::CmpIOp>(loc, predicate, promoted[0], promoted[1])
        .getResult();
  }
};

template <arith::CmpIPredicate signedPred, arith::CmpIPredicate unsignedPred>
struct CmpIntInequalityOp {
  Value create(ConversionPatternRewriter &rewriter, Location loc,
               Type resultType, ArrayRef<Value> operands,
               ArrayRef<Type> types) {
    SmallVector<Value, 2> promoted;
    auto bestType = promoteTypes(rewriter, loc, operands, types, &promoted);
    auto predicate = bestType.isSignedInteger() ? signedPred : unsignedPred;
    return rewriter
        .create<mlir::arith::CmpIOp>(loc, predicate, promoted[0], promoted[1])
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
            rewriter.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
        promoted.push_back(
            rewriter
                .create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, operand, zero)
                .getResult());
      } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
        auto zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
        promoted.push_back(
            rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ne, operand, zero)
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
      auto zero = rewriter.create<mlir::arith::ConstantFloatOp>(loc, value, floatType);
      return rewriter.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, input, zero)
          .getResult();
    } else if (auto intType = fromType.dyn_cast<IntegerType>()) {
      auto zero = rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
      return rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::eq, input, zero)
          .getResult();
    } else {
      llvm_unreachable("Unknown type for LogicalNotOp");
    }
  }
};

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

static arith::AtomicRMWKind convertAgg(AggregationKind agg, Type type) {
  switch (agg) {
  case AggregationKind::assign:
    return arith::AtomicRMWKind::assign;
  case AggregationKind::add:
    if (type.isa<FloatType>()) {
      return arith::AtomicRMWKind::addf;
    } else {
      return arith::AtomicRMWKind::addi;
    }
  case AggregationKind::mul:
    if (type.isa<FloatType>()) {
      return arith::AtomicRMWKind::mulf;
    } else {
      return arith::AtomicRMWKind::muli;
    }
  case AggregationKind::min:
    if (type.isa<FloatType>()) {
      return arith::AtomicRMWKind::minf;
    } else if (type.isSignedInteger()) {
      return arith::AtomicRMWKind::mins;
    } else {
      return arith::AtomicRMWKind::minu;
    }
  case AggregationKind::max:
    if (type.isa<FloatType>()) {
      return arith::AtomicRMWKind::maxf;
    } else if (type.isSignedInteger()) {
      return arith::AtomicRMWKind::maxs;
    } else {
      return arith::AtomicRMWKind::maxu;
    }
  }
  llvm_unreachable("Invalid agg type in convertAgg");
}

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
    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
        /*ranges=*/alloc.rankedTensorType.getShape());
    if (Attribute attr = op->getAttr("name"))
      forOp->setAttr("name", attr);
    if (Attribute attr = op->getAttr("schedule"))
      forOp->setAttr("schedule", attr);

    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the loads
    SmallVector<Value, 4> scalars;
    for (size_t i = 0; i < operands.size(); i++) {
      auto maybePadding =
          tile::getPaddingInfo(op->getOperand(i).getDefiningOp());
      scalars.push_back(buildBroadcastLoad(rewriter, loc, operands[i],
                                           alloc.memRefType.getRank(),
                                           maybePadding));
    }

    // Create the standard op
    SmallVector<Type, 4> operandTypes;
    for (auto type : op->getOperandTypes()) {
      operandTypes.push_back(getElementType(type));
    }
    IntoOpBuilder intoOpBuilder;
    auto result = intoOpBuilder.create(rewriter, loc, alloc.elementType,
                                       scalars, operandTypes);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, alloc.resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace output with the newly allocated buffer
    rewriter.replaceOp(op, forOp.getResult(0));
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

    if (auto attr = op->getAttrOfType<StringAttr>("trace")) {
      auto module = op->getParentOfType<ModuleOp>();
      auto symbol = createStubTraceFunc(module, attr);
      rewriter.create<CallOp>(loc, symbol, ArrayRef<Type>{});
    }

    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Do initialization
    ArrayRef<int64_t> shape = alloc.rankedTensorType.getShape();
    auto parallel = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
        /*ranges=*/shape);
    auto parallelBuilder = parallel.getBodyBuilder();
    auto maybePadding = tile::getPaddingInfo(op.init().getDefiningOp());
    auto load = buildBroadcastLoad(parallelBuilder, loc, cionAdaptor.init(),
                                   shape.size(), maybePadding);
    auto store = buildSimpleStore(parallelBuilder, loc, load,
                                  alloc.resultMemRef, tile::getPaddingInfo(op));
    if (maybePadding)
      updateAffineMap(store.getDefiningOp(), *maybePadding);
    parallelBuilder.create<AffineYieldOp>(loc, store);
    auto filled = parallel.getResult(0);

    // Determine lower and upper bounds.
    SmallVector<AffineMap, 8> lbMaps;
    SmallVector<AffineMap, 8> ubMaps;
    auto lowerBounds = op.lowerBounds().getValue();
    auto upperBounds = op.upperBounds().getValue();
    assert(lowerBounds.getNumResults() == upperBounds.getNumResults() &&
           "mismatched dims for lower and upper bounds");
    for (unsigned i = 0; i < lowerBounds.getNumResults(); i++) {
      AffineExpr ubExpr = upperBounds.getResult(i) + 1;
      int64_t upper = ubExpr.cast<AffineConstantExpr>().getValue();
      lbMaps.push_back(AffineMap::get(lowerBounds.getNumDims(),
                                      lowerBounds.getNumSymbols(),
                                      lowerBounds.getResult(i)));
      ubMaps.push_back(AffineMap::getConstantMap(upper, op.getContext()));
    }

    // Make the outer loops
    SmallVector<int64_t, 8> steps(lbMaps.size(), 1);
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
        /*lbMaps=*/lbMaps,
        /*lbArgs=*/ArrayRef<Value>{},
        /*ubMaps=*/ubMaps,
        /*ubArgs=*/ArrayRef<Value>{},
        /*steps=*/steps);
    if (Attribute attr = op->getAttr("name"))
      forOp->setAttr("name", attr);
    if (Attribute attr = op->getAttr("schedule"))
      forOp->setAttr("schedule", attr);

    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);
    auto idxs = body->getArguments();

    // add constraints
    if (op.cons()) {
      auto cons = op.cons().getValue();
      auto ifOp = rewriter.create<AffineIfOp>(loc, TypeRange{alloc.memRefType},
                                              cons, idxs, true);
      rewriter.create<AffineYieldOp>(loc, ifOp->getResults());
      rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
      rewriter.create<AffineYieldOp>(loc, filled);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());
    }

    // Create the loads + casts
    SmallVector<Value, 4> scalars;
    auto srcs = op.srcs().getValue();
    for (size_t i = 0; i < srcs.size(); i++) {
      auto operand = cionOperands[i];
      if (!operand.getType().isa<MemRefType>()) {
        scalars.push_back(operand);
      } else {
        auto map = srcs[i].cast<AffineMapAttr>().getValue();
        auto loadOp = rewriter.create<pxa::PxaLoadOp>(loc, operand, map, idxs);
        auto maybePadding =
            tile::getPaddingInfo(op.operands()[i].getDefiningOp());
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
    maybePadding = tile::getPaddingInfo(op);
    if (maybePadding)
      updateAffineMap(reduceOp, *maybePadding);
    rewriter.create<AffineYieldOp>(loc, ValueRange{reduceOp});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));
  }
};

struct IndexOpConversion : public OpConversionPattern<tile::IndexOp> {
  using OpConversionPattern<tile::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Gather some basic info
    auto loc = op.getLoc();
    TileToPXATypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto resultMemRef =
        rewriter.create<memref::AllocOp>(loc, resultType).getResult();

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{resultType},
        /*reductions=*/ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
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
    auto cast = rewriter.create<mlir::arith::IndexCastOp>(loc, apply,
                                                   rewriter.getIntegerType(32));
    auto stored = buildSimpleStore(rewriter, loc, cast, resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<tile::ReshapeOp> {
  using OpConversionPattern<tile::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ReshapeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create an adaptor, to interpret the operands
    tile::ReshapeOpAdaptor adaptor(operands);

    auto tensor = adaptor.tensor();

    TileToPXATypeConverter typeConverter;
    auto resultType = typeConverter.convertType(op.result().getType());

    rewriter.replaceOpWithNewOp<stdx::ReshapeOp>(op, resultType, tensor);
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
    TileToPXATypeConverter typeConverter;
    auto resultType =
        typeConverter.convertType(op.result().getType()).cast<MemRefType>();

    // Make an allocation for the output
    auto memRef = rewriter.create<memref::AllocOp>(loc, resultType).getResult();

    // Populate the buffer with the shape dims
    auto operandType = adaptor.tensor().getType().cast<MemRefType>();
    auto aggOp = arith::AtomicRMWKind::assign;
    for (unsigned i = 0; i < operandType.getRank(); i++) {
      auto dim = rewriter.create<mlir::memref::DimOp>(loc, adaptor.tensor(), i);
      auto cast = rewriter.create<mlir::arith::IndexCastOp>(
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

struct CastOpConversion : public OpConversionPattern<tile::CastOp> {
  using OpConversionPattern<tile::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::CastOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    BufferAllocator alloc(rewriter, op.getOperation(), op.result().getType());

    // Make a parallel for loop to fill the result
    auto forOp = rewriter.create<AffineParallelOp>(
        loc,
        /*resultTypes=*/ArrayRef<Type>{alloc.memRefType},
        /*reductions=*/ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
        /*ranges=*/alloc.rankedTensorType.getShape());
    auto body = forOp.getBody();
    rewriter.setInsertionPointToStart(body);

    // Create the load
    auto scalar = buildBroadcastLoad(rewriter, loc, operands[0],
                                     alloc.memRefType.getRank());

    // Create the standard cast op
    auto dtype = getElementType(op.tensor());
    bool resultIsSigned =
        getElementType(op.result().getType()).isSignedInteger();
    auto result = createCastOp(rewriter, loc, scalar, dtype.isSignedInteger(),
                               alloc.elementType, resultIsSigned);

    // Create the store
    auto stored = buildSimpleStore(rewriter, loc, result, alloc.resultMemRef,
                                   tile::getPaddingInfo(op));
    rewriter.create<AffineYieldOp>(loc, ValueRange{stored});

    // Replace the op
    rewriter.replaceOp(op, forOp.getResult(0));

    return success();
  }
};

template <typename FuncLikeOp>
struct FuncOpConversion : public OpConversionPattern<FuncLikeOp> {
  using OpConversionPattern<FuncLikeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncLikeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();

    // Convert the function signature
    TileToPXATypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
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
    NamedAttribute msg = op.attrs().getNamed("msg");
    if (!msg) {
      return failure();
    }
    auto symbol = createStubTraceFunc(module, msg->getValue().cast<StringAttr>());
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

struct LowerTileToPXAPass : public LowerTileToPXABase<LowerTileToPXAPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    // Inject tile.ident ops for each return operand that needs it.
    // argument, a constant value, or a reshape op.
    auto injectIdent = [](Operation *op) {
      OpBuilder builder(op);
      for (OpOperand &operand : op->getOpOperands()) {
        Value value = operand.get();
        Value def = pxa::getIndirectDef(value);
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
    TileToPXATypeConverter converter;
    target.addLegalDialect<mlir::AffineDialect,         //
                           mlir::StandardOpsDialect,    //
                           mlir::math::MathDialect,     //
                           mlir::memref::MemRefDialect, //
                           mlir::scf::SCFDialect,       //
                           layer::LayerDialect,         //
                           pxa::PXADialect,             //
                           arith::ArithmeticDialect,    //
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
    target.addDynamicallyLegalOp<scf::ForOp>(
        [&](scf::ForOp op) { return converter.isLegal(op.getResultTypes()); });

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
        CastOpConversion,                  //
        ConstantOpConversion,              //
        FuncOpConversion<FuncOp>,          //
        FuncOpConversion<stdx::ClosureOp>, //
        IndexOpConversion,                 //
        PragmaOpConversion,                //
        ReshapeOpConversion,               //
        ShapeOpConversion,                 //
        TraceOpConversion,                 //
        ScfForOpConversion,                //
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
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpFloatOp<arith::CmpFPredicate::OEQ>>,
                                AnyComparandIs<EltwiseFloat>>,
        ContractionOpConversion<CombinationKind::cond,
                                CondOp<CmpIntOp<arith::CmpIPredicate::eq>>,
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
        EltwiseOpConversion<tile::SelectOp, SelectOpBuilder>,
        EltwiseOpConversion<tile::IdentOp, FirstOperand>>(&getContext());

    populateTileToPXASpecialPatterns(patterns);

    // Run the conversion
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      for (ReturnOp returnOp : funcOp.getOps<ReturnOp>()) {
        connectResults(funcOp, returnOp);
        for (stdx::ClosureOp closureOp : funcOp.getOps<stdx::ClosureOp>()) {
          for (stdx::YieldOp yieldOp : closureOp.getOps<stdx::YieldOp>()) {
            connectResults(closureOp, yieldOp);
          }
        }
      }
    }
  }

  template <typename FuncLikeOp, typename ReturnLikeOp>
  void connectResults(FuncLikeOp funcOp, ReturnLikeOp returnOp) {
    unsigned argNumber =
        funcOp.getType().getNumInputs() - returnOp.getNumOperands();
    for (Value operand : returnOp.operands()) {
      // Find very initial allocation of memref
      Value def = pxa::getIndirectDef(operand);
      BlockArgument blockArg = funcOp.getBody().getArgument(argNumber++);
      if (def != blockArg) {
        def.replaceAllUsesWith(blockArg);
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerTileToPXAPass() {
  return std::make_unique<LowerTileToPXAPass>();
}

} // namespace pmlc::conversion::tile_to_pxa
