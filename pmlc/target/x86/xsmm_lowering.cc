// Copyright 2020 Intel Corporation

#include "pmlc/target/x86/xsmm_lowering.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/xsmm/ir/dialect.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace xsmm = dialect::xsmm;

namespace {

/// Visit affine expressions recursively and build the sequence of operations
/// that correspond to it.  Visitation functions return an Value of the
/// expression subtree they visited or `nullptr` on error.
class AffineApplyExpander
    : public AffineExprVisitor<AffineApplyExpander, Value> {
public:
  /// This internal class expects arguments to be non-null, checks must be
  /// performed at the call site.
  AffineApplyExpander(OpBuilder &builder, ValueRange dimValues,
                      ValueRange symbolValues, Location loc)
      : builder(builder), dimValues(dimValues), symbolValues(symbolValues),
        loc(loc) {}

  template <typename OpTy>
  Value buildBinaryExpr(AffineBinaryOpExpr expr) {
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    if (!lhs || !rhs)
      return nullptr;
    auto op = builder.create<OpTy>(loc, lhs, rhs);
    return op.getResult();
  }

  Value visitAddExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<AddIOp>(expr);
  }

  Value visitMulExpr(AffineBinaryOpExpr expr) {
    return buildBinaryExpr<MulIOp>(expr);
  }

  /// Euclidean modulo operation: negative RHS is not allowed.
  /// Remainder of the euclidean integer division is always non-negative.
  ///
  /// Implemented as
  ///
  ///     a mod b =
  ///         let remainder = srem a, b;
  ///             negative = a < 0 in
  ///         select negative, remainder + b, remainder.
  Value visitModExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (modulo by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "modulo by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value remainder = builder.create<SignedRemIOp>(loc, lhs, rhs);
    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value isRemainderNegative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, remainder, zeroCst);
    Value correctedRemainder = builder.create<AddIOp>(loc, remainder, rhs);
    Value result = builder.create<SelectOp>(loc, isRemainderNegative,
                                            correctedRemainder, remainder);
    return result;
  }

  /// Floor division operation (rounds towards negative infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///        a floordiv b =
  ///            let negative = a < 0 in
  ///            let absolute = negative ? -a - 1 : a in
  ///            let quotient = absolute / b in
  ///                negative ? -quotient - 1 : quotient
  Value visitFloorDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(
          loc,
          "semi-affine expressions (division by non-const) are not supported");
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }

    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value noneCst = builder.create<ConstantIndexOp>(loc, -1);
    Value negative =
        builder.create<CmpIOp>(loc, CmpIPredicate::slt, lhs, zeroCst);
    Value negatedDecremented = builder.create<SubIOp>(loc, noneCst, lhs);
    Value dividend =
        builder.create<SelectOp>(loc, negative, negatedDecremented, lhs);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value correctedQuotient = builder.create<SubIOp>(loc, noneCst, quotient);
    Value result =
        builder.create<SelectOp>(loc, negative, correctedQuotient, quotient);
    return result;
  }

  /// Ceiling division operation (rounds towards positive infinity).
  ///
  /// For positive divisors, it can be implemented without branching and with a
  /// single division operation as
  ///
  ///     a ceildiv b =
  ///         let negative = a <= 0 in
  ///         let absolute = negative ? -a : a - 1 in
  ///         let quotient = absolute / b in
  ///             negative ? -quotient : quotient + 1
  Value visitCeilDivExpr(AffineBinaryOpExpr expr) {
    auto rhsConst = expr.getRHS().dyn_cast<AffineConstantExpr>();
    if (!rhsConst) {
      emitError(loc) << "semi-affine expressions (division by non-const) are "
                        "not supported";
      return nullptr;
    }
    if (rhsConst.getValue() <= 0) {
      emitError(loc, "division by non-positive value is not supported");
      return nullptr;
    }
    auto lhs = visit(expr.getLHS());
    auto rhs = visit(expr.getRHS());
    assert(lhs && rhs && "unexpected affine expr lowering failure");

    Value zeroCst = builder.create<ConstantIndexOp>(loc, 0);
    Value oneCst = builder.create<ConstantIndexOp>(loc, 1);
    Value nonPositive =
        builder.create<CmpIOp>(loc, CmpIPredicate::sle, lhs, zeroCst);
    Value negated = builder.create<SubIOp>(loc, zeroCst, lhs);
    Value decremented = builder.create<SubIOp>(loc, lhs, oneCst);
    Value dividend =
        builder.create<SelectOp>(loc, nonPositive, negated, decremented);
    Value quotient = builder.create<SignedDivIOp>(loc, dividend, rhs);
    Value negatedQuotient = builder.create<SubIOp>(loc, zeroCst, quotient);
    Value incrementedQuotient = builder.create<AddIOp>(loc, quotient, oneCst);
    Value result = builder.create<SelectOp>(loc, nonPositive, negatedQuotient,
                                            incrementedQuotient);
    return result;
  }

  Value visitConstantExpr(AffineConstantExpr expr) {
    auto valueAttr =
        builder.getIntegerAttr(builder.getIndexType(), expr.getValue());
    auto op =
        builder.create<ConstantOp>(loc, builder.getIndexType(), valueAttr);
    return op.getResult();
  }

  Value visitDimExpr(AffineDimExpr expr) {
    assert(expr.getPosition() < dimValues.size() &&
           "affine dim position out of range");
    return dimValues[expr.getPosition()];
  }

  Value visitSymbolExpr(AffineSymbolExpr expr) {
    assert(expr.getPosition() < symbolValues.size() &&
           "symbol dim position out of range");
    return symbolValues[expr.getPosition()];
  }

private:
  OpBuilder &builder;
  ValueRange dimValues;
  ValueRange symbolValues;

  Location loc;
};

/// Create a sequence of operations that implement the `expr` applied to the
/// given dimension and symbol values.
Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                       ValueRange dimValues, ValueRange symbolValues) {
  return AffineApplyExpander(builder, dimValues, symbolValues, loc).visit(expr);
}

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value, 8>> static expandAffineMap(OpBuilder &builder,
                                                       Location loc,
                                                       AffineMap affineMap,
                                                       ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, &builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return None;
}

class XSMMGemmLowering : public OpRewritePattern<xsmm::GemmOp> {
public:
  using OpRewritePattern<xsmm::GemmOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(xsmm::GemmOp op,
                                     PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto symbol = impl.getOrInsertFunc();
    auto a = impl.prepareOperand(op.a(), op.aMap(), op.getOperandsForA());
    auto b = impl.prepareOperand(op.b(), op.bMap(), op.getOperandsForB());
    auto c = impl.prepareOperand(op.c(), op.cMap(), op.getOperandsForC());
    auto lda = impl.createConstantIntOp(op.lda().getSExtValue());
    auto ldb = impl.createConstantIntOp(op.ldb().getSExtValue());
    auto ldc = impl.createConstantIntOp(op.ldc().getSExtValue());
    SmallVector<Value, 9> args{a, b, c, lda, ldb, ldc};
    for (auto attr : op.tile().getValue()) {
      auto intValue = attr.cast<IntegerAttr>().getInt();
      auto value = impl.createConstantIntOp(intValue);
      args.push_back(value);
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, symbol, ArrayRef<Type>{}, args);
    // auto call =
    //     rewriter.create<CallOp>(impl.loc, symbol, ArrayRef<Type>{}, args);
    // IVLOG(1, "here: " << mlir::debugString(*call));
    // rewriter.eraseOp(op);
    return matchSuccess();
  }

  struct Impl {
    xsmm::GemmOp op;
    PatternRewriter &rewriter;
    Location loc;
    ModuleOp module;
    Type i32Type;
    Type elementType;
    UnrankedMemRefType memRefType;

    Impl(xsmm::GemmOp op, PatternRewriter &rewriter)
        : op(op), rewriter(rewriter), loc(op.getLoc()),
          module(op.getParentOfType<ModuleOp>()),
          i32Type(rewriter.getIntegerType(32)),
          elementType(rewriter.getF32Type()),
          memRefType(UnrankedMemRefType::get(elementType, /*memorySpace=*/0)) {}

    FlatSymbolRefAttr getOrInsertFunc() {
      const char *symbol = "plaidml_rt_xsmm_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      std::array<Type, 9> inputs{memRefType, memRefType, memRefType,
                                 i32Type,    i32Type,    i32Type,
                                 i32Type,    i32Type,    i32Type};
      ArrayRef<Type> results{};
      auto funcType = rewriter.getFunctionType(inputs, results);
      ArrayRef<NamedAttribute> attrs{};
      rewriter.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
    }

    Value prepareOperand(Value operand, AffineMap map, ValueRange mapOperands) {
      auto resultOperands = expandAffineMap(rewriter, loc, map, mapOperands);
      ArrayRef<Value> empty{};
      auto subview = rewriter.create<SubViewOp>(loc, operand, *resultOperands,
                                                empty, empty);
      return rewriter.create<MemRefCastOp>(loc, subview, memRefType);
    }

    Value createConstantIntOp(int64_t value) {
      return rewriter.create<ConstantIntOp>(loc, value, i32Type);
    }
  };
};

} // namespace

void populateXSMMConversionPatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *ctx) {
  patterns.insert<XSMMGemmLowering>(ctx);
}

namespace {

class LowerXSMMPass : public FunctionPass<LowerXSMMPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateXSMMConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
    // target.addIllegalDialect<xsmm::Dialect>();
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<LowerXSMMPass> pass("xsmm",
                                            "XSMM to standard conversion");

} // namespace pmlc::target::x86
