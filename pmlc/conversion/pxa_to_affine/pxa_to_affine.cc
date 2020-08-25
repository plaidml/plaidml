// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/pxa_to_affine/pass_detail.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::conversion::pxa_to_affine {

namespace pxa = dialect::pxa;

namespace {

struct AffineParallelOpConversion
    : public OpConversionPattern<AffineParallelOp> {
  using OpConversionPattern<AffineParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineParallelOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // This conversion doesn't work in the rank 0 case; that case will be
    // covered by canonicalization.
    if (op.lowerBoundsMap().getNumResults() == 0)
      return failure();
    // Create an affine loop nest, capture induction variables
    llvm::SmallVector<Value, 8> ivs;
    auto steps = op.getSteps();
    for (unsigned int i = 0; i < op.lowerBoundsMap().getNumResults(); i++) {
      auto step = steps[i];
      auto forOp = rewriter.create<AffineForOp>(
          op.getLoc(), op.getLowerBoundsOperands(),
          op.lowerBoundsMap().getSubMap({i}), op.getUpperBoundsOperands(),
          op.upperBoundsMap().getSubMap({i}), step);
      rewriter.setInsertionPointToStart(&forOp.region().front());
      ivs.push_back(forOp.getInductionVar());
    }
    // Move ParallelOp's operations (single block) to Affine innermost loop.
    auto &innerLoopOps = rewriter.getInsertionBlock()->getOperations();
    auto &parallelBodyOps = op.region().front().getOperations();
    innerLoopOps.splice(std::prev(innerLoopOps.end()), parallelBodyOps,
                        parallelBodyOps.begin(),
                        std::prev(parallelBodyOps.end()));
    // Replace all uses of old values
    size_t idx = 0;
    for (auto arg : op.region().front().getArguments()) {
      arg.replaceAllUsesWith(ivs[idx++]);
    }
    // Replace outputs with values from yield
    auto termIt = std::prev(parallelBodyOps.end());
    for (size_t i = 0; i < op.getNumResults(); i++) {
      op.getResult(i).replaceAllUsesWith(termIt->getOperand(i));
    }
    // We are done. Remove original op.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AffineIfOpConversion : public OpConversionPattern<AffineIfOp> {
  using OpConversionPattern<AffineIfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineIfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // Make a new if value
    auto newIf = rewriter.create<AffineIfOp>(op.getLoc(), op.getIntegerSet(),
                                             op.getOperands(), op.hasElse());
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
    return success();
  }
};

struct PxaLoadOpConversion : public OpConversionPattern<pxa::PxaLoadOp> {
  using OpConversionPattern<pxa::PxaLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::PxaLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AffineLoadOp>(op, op.memref(),
                                              op.getAffineMap(), op.indices());
    return success();
  }
};

struct PxaVectorLoadOpConversion
    : public OpConversionPattern<pxa::PxaVectorLoadOp> {
  using OpConversionPattern<pxa::PxaVectorLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::PxaVectorLoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AffineVectorLoadOp>(
        op, op.getVectorType(), op.memref(), op.getAffineMap(), op.indices());
    return success();
  }
};

static Value createReduction(ConversionPatternRewriter &rewriter, Location loc,
                             AtomicRMWKind agg, Value source, Value val) {
  switch (agg) {
  case AtomicRMWKind::assign:
    return val;
  case AtomicRMWKind::addf:
    return rewriter.create<AddFOp>(loc, source, val);
  case AtomicRMWKind::addi:
    return rewriter.create<AddIOp>(loc, source, val);
  case AtomicRMWKind::maxf: {
    auto cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::maxu: {
    auto cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::ugt, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::maxs: {
    auto cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::minf: {
    auto cmp = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::minu: {
    auto cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::ult, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::mins: {
    auto cmp = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, val, source);
    return rewriter.create<SelectOp>(loc, cmp, val, source);
  }
  case AtomicRMWKind::mulf:
    return rewriter.create<MulFOp>(loc, source, val);
  case AtomicRMWKind::muli:
    return rewriter.create<MulIOp>(loc, source, val);
  default:
    llvm_unreachable("Unsupported aggregation for "
                     "PxaReduceOpConversion::createReduction");
  }
}

struct PxaReduceOpConversion : public OpConversionPattern<pxa::PxaReduceOp> {
  using OpConversionPattern<pxa::PxaReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::PxaReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto source = rewriter.create<AffineLoadOp>(op.getLoc(), op.memref(),
                                                op.map(), op.idxs());
    auto reduce =
        createReduction(rewriter, op.getLoc(), op.agg(), source, op.val());
    rewriter.create<AffineStoreOp>(op.getLoc(), reduce, op.memref(), op.map(),
                                   op.idxs());
    op.replaceAllUsesWith(op.memref());
    rewriter.eraseOp(op);
    return success();
  }
};

struct PxaVectorReduceOpConversion
    : public OpConversionPattern<pxa::PxaVectorReduceOp> {
  using OpConversionPattern<pxa::PxaVectorReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::PxaVectorReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto source = rewriter.create<AffineVectorLoadOp>(
        op.getLoc(), op.getVectorType(), op.memref(), op.getAffineMap(),
        op.idxs());
    auto reduce =
        createReduction(rewriter, op.getLoc(), op.agg(), source, op.vector());
    rewriter.create<AffineVectorStoreOp>(op.getLoc(), reduce, op.memref(),
                                         op.getAffineMap(), op.idxs());
    op.replaceAllUsesWith(op.memref());
    rewriter.eraseOp(op);
    return success();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();
    if (op.isExternal()) {
      return success();
    }
    IVLOG(2, "FuncOpConversion::rewrite> " << debugString(type));

    // Convert the function signature
    TypeConverter::SignatureConversion result(type.getNumInputs() +
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
      operand.replaceAllUsesWith(block.getArgument(blockArg++));
    }
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return success();
  }
};

struct LowerPXAToAffinePass
    : public LowerPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    PXAToAffineConversionTarget target(ctx);

    OwningRewritePatternList patterns;
    populatePXAToAffineConversionPatterns(patterns, &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

} // namespace

PXAToAffineConversionTarget::PXAToAffineConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<AffineDialect>();
  addLegalDialect<StandardOpsDialect>();
  addIllegalDialect<pxa::PXADialect>();
  addIllegalOp<AffineParallelOp>();
  addDynamicallyLegalOp<AffineIfOp>(
      [](AffineIfOp op) { return op.getNumResults() == 0; });
  addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
    return op.isExternal() || op.getType().getNumResults() == 0;
  });
  addDynamicallyLegalOp<ReturnOp>(
      [](ReturnOp op) { return op.getNumOperands() == 0; });
}

void populatePXAToAffineConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  patterns.insert<                 //
      AffineIfOpConversion,        //
      AffineParallelOpConversion,  //
      FuncOpConversion,            //
      PxaLoadOpConversion,         //
      PxaReduceOpConversion,       //
      PxaVectorLoadOpConversion,   //
      PxaVectorReduceOpConversion, //
      ReturnOpConversion>(ctx);
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

} // namespace pmlc::conversion::pxa_to_affine
