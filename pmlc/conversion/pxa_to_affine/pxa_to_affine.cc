// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/pxa_to_affine/pass_detail.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::pxa_to_affine {
namespace pxa = dialect::pxa;

using mlir::AffineIfOp;
using mlir::AffineLoadOp;
using mlir::AffineParallelOp;
using mlir::AffineStoreOp;
using mlir::AllocOp;
using mlir::ArrayRef;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::RankedTensorType;
using mlir::ReturnOp;
using mlir::Type;
using mlir::Value;

using util::AggregationKind;

namespace {

struct LowerPXAToAffinePass
    : public LowerPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final;
};

template <typename OpType>
struct LoweringBase : public OpConversionPattern<OpType> {
  MLIRContext *ctx;

  explicit LoweringBase(MLIRContext *ctx)
      : OpConversionPattern<OpType>(ctx), ctx(ctx) {}
  LogicalResult match(Operation *op) const override { return mlir::success(); }
};

struct AffineParallelOpConversion : public LoweringBase<AffineParallelOp> {
  explicit AffineParallelOpConversion(MLIRContext *ctx) : LoweringBase(ctx) {}

  void rewrite(AffineParallelOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    // Create an affine loop nest, capture induction variables
    llvm::SmallVector<Value, 8> ivs;
    for (unsigned int i = 0; i < op.lowerBoundsMap().getNumResults(); i++) {
      auto step = op.steps().getValue()[i].cast<IntegerAttr>().getInt();
      auto af = rewriter.create<mlir::AffineForOp>(
          op.getLoc(), op.getLowerBoundsOperands(),
          op.lowerBoundsMap().getSubMap({i}), op.getUpperBoundsOperands(),
          op.upperBoundsMap().getSubMap({i}), step);
      rewriter.setInsertionPointToStart(&af.region().front());
      ivs.push_back(af.getInductionVar());
    }
    // Move ParallelOp's operations (single block) to Affine innermost loop.
    auto &innerLoopOps = rewriter.getInsertionBlock()->getOperations();
    auto &stripeBodyOps = op.region().front().getOperations();
    innerLoopOps.splice(std::prev(innerLoopOps.end()), stripeBodyOps,
                        stripeBodyOps.begin(), std::prev(stripeBodyOps.end()));
    // Replace all uses of old values
    size_t idx = 0;
    for (auto arg : op.region().front().getArguments()) {
      arg.replaceAllUsesWith(ivs[idx++]);
    }
    // Replace outputs with values from yield
    auto termIt = std::prev(stripeBodyOps.end());
    for (size_t i = 0; i < op.getNumResults(); i++) {
      op.getResult(i).replaceAllUsesWith(termIt->getOperand(i));
    }
    // We are done. Remove original op.
    rewriter.eraseOp(op);
  }
};

struct AffineIfOpConversion : public LoweringBase<AffineIfOp> {
  explicit AffineIfOpConversion(MLIRContext *ctx) : LoweringBase(ctx) {}

  void rewrite(AffineIfOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    // Make a new if value
    auto newIf = rewriter.create<mlir::AffineIfOp>(
        op.getLoc(), op.getIntegerSet(), op.getOperands(), op.hasElse());
    // Move 'then' operations over, ignoring terminator
    auto &newThenOps = newIf.getThenBlock()->getOperations();
    auto &oldThenOps = op.getThenBlock()->getOperations();
    newThenOps.splice(std::prev(newThenOps.end()), oldThenOps,
                      oldThenOps.begin(), std::prev(oldThenOps.end()));
    // Replace outputs with values from yield (based on the then clause)
    auto termIt = std::prev(oldThenOps.end());
    for (size_t i = 0; i < op.getNumResults(); i++) {
      op.getResult(i).replaceAllUsesWith(termIt->getOperand(i));
    }
    // Move 'else' operations over, ignoring terminator
    auto &newElseOps = newIf.getElseBlock()->getOperations();
    auto &oldElseOps = op.getElseBlock()->getOperations();
    newElseOps.splice(std::prev(newElseOps.end()), oldElseOps,
                      oldElseOps.begin(), std::prev(oldElseOps.end()));
    // Erase original
    rewriter.eraseOp(op);
  }
};

struct AffineReduceOpConversion : public LoweringBase<pxa::AffineReduceOp> {
  explicit AffineReduceOpConversion(MLIRContext *ctx) : LoweringBase(ctx) {}

  void rewrite(pxa::AffineReduceOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto source = rewriter.create<AffineLoadOp>(op.getLoc(), op.mem(), op.map(),
                                                op.idxs());
    auto reduce = createReduction(rewriter, op, source.getResult());
    rewriter.create<AffineStoreOp>(op.getLoc(), reduce, op.mem(), op.map(),
                                   op.idxs());
    op.replaceAllUsesWith(op.mem());
    rewriter.eraseOp(op);
  }

  Value createReduction(ConversionPatternRewriter &rewriter,
                        pxa::AffineReduceOp op, Value source) const {
    switch (op.agg()) {
    case AggregationKind::assign:
      return op.val();
    case AggregationKind::add: {
      if (source.getType().isa<FloatType>()) {
        return rewriter.create<mlir::AddFOp>(op.getLoc(), source, op.val());
      }
      return rewriter.create<mlir::AddIOp>(op.getLoc(), source, op.val());
    }
    case AggregationKind::max: {
      if (source.getType().isa<FloatType>()) {
        auto cmp = rewriter.create<mlir::CmpFOp>(
            op.getLoc(), mlir::CmpFPredicate::OGT, op.val(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(),
                                               source);
      }
      // TODO: determine whether to use signed or unsigned compare
      auto cmp = rewriter.create<mlir::CmpIOp>(
          op.getLoc(), mlir::CmpIPredicate::sgt, op.val(), source);
      return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(),
                                             source);
    }
    case AggregationKind::min: {
      if (source.getType().isa<FloatType>()) {
        auto cmp = rewriter.create<mlir::CmpFOp>(
            op.getLoc(), mlir::CmpFPredicate::OLT, op.val(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(),
                                               source);
      }
      // TODO: determine whether to use signed or unsigned compare
      auto cmp = rewriter.create<mlir::CmpIOp>(
          op.getLoc(), mlir::CmpIPredicate::slt, op.val(), source);
      return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(),
                                             source);
    }
    case AggregationKind::mul: {
      if (source.getType().isa<FloatType>()) {
        return rewriter.create<mlir::MulFOp>(op.getLoc(), source, op.val());
      }
      return rewriter.create<mlir::MulIOp>(op.getLoc(), source, op.val());
    }
    default:
      llvm_unreachable("Unsupported aggregation for "
                       "AffineReduceOpConversion::createReduction");
    }
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();
    IVLOG(2, "FuncOpConversion::rewrite> " << mlir::debugString(type));

    // Convert the function signature
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() +
                                                    type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {type.getInput(i)});
    }
    for (unsigned i = 0; i < type.getNumResults(); ++i) {
      result.addInputs({type.getResult(i)});
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

    return mlir::success();
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
      operand.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return mlir::success();
  }
};

void LowerPXAToAffinePass::runOnOperation() {
  // Set up target (i.e. what is legal)
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<pxa::PXADialect>();
  target.addIllegalOp<AffineParallelOp>();
  target.addDynamicallyLegalOp<AffineIfOp>(
      [](AffineIfOp op) { return op.getNumResults() == 0; });
  target.addDynamicallyLegalOp<FuncOp>(
      [](FuncOp op) { return op.getType().getNumResults() == 0; });
  target.addDynamicallyLegalOp<ReturnOp>(
      [](ReturnOp op) { return op.getNumOperands() == 0; });

  // Setup rewrite patterns
  mlir::OwningRewritePatternList patterns;
  patterns
      .insert<AffineParallelOpConversion, AffineIfOpConversion,
              AffineReduceOpConversion, FuncOpConversion, ReturnOpConversion>(
          &getContext());

  // Run the conversion
  if (failed(
          applyPartialConversion(getOperation(), target, patterns, nullptr))) {
    getOperation().dump();
    emitError(mlir::UnknownLoc::get(&getContext()),
              "Error lowering pxa -> affine\n");
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

} // namespace pmlc::conversion::pxa_to_affine
