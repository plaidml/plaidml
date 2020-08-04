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

namespace pmlc::conversion::pxa_to_affine {
namespace pxa = dialect::pxa;

using mlir::AffineIfOp;
using mlir::AffineLoadOp;
using mlir::AffineMapAttr;
using mlir::AffineParallelOp;
using mlir::AffineStoreOp;
using mlir::AffineVectorLoadOp;
using mlir::AffineVectorStoreOp;
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
using mlir::VectorType;

using util::AggregationKind;

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
      return mlir::failure();
    // Create an affine loop nest, capture induction variables
    llvm::SmallVector<Value, 8> ivs;
    for (unsigned int i = 0; i < op.lowerBoundsMap().getNumResults(); i++) {
      auto step = op.steps().getValue()[i].cast<IntegerAttr>().getInt();
      auto forOp = rewriter.create<mlir::AffineForOp>(
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
    return mlir::success();
  }
};

struct AffineIfOpConversion : public OpConversionPattern<AffineIfOp> {
  using OpConversionPattern<AffineIfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineIfOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
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
    return mlir::success();
  }
};

struct AffineReduceOpConversion
    : public OpConversionPattern<pxa::AffineReduceOp> {
  using OpConversionPattern<pxa::AffineReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::AffineReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto source = rewriter.create<AffineLoadOp>(op.getLoc(), op.mem(), op.map(),
                                                op.idxs());
    auto reduce = createReduction(rewriter, op, source.getResult());
    rewriter.create<AffineStoreOp>(op.getLoc(), reduce, op.mem(), op.map(),
                                   op.idxs());
    op.replaceAllUsesWith(op.mem());
    rewriter.eraseOp(op);
    return mlir::success();
  }

  Value createReduction(ConversionPatternRewriter &rewriter,
                        pxa::AffineReduceOp op, Value source) const {
    auto type = source.getType();
    switch (op.agg()) {
    case AggregationKind::assign:
      return op.val();
    case AggregationKind::add: {
      if (type.isa<FloatType>()) {
        return rewriter.create<mlir::AddFOp>(op.getLoc(), source, op.val());
      }
      return rewriter.create<mlir::AddIOp>(op.getLoc(), source, op.val());
    }
    case AggregationKind::max: {
      if (type.isa<FloatType>()) {
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
      if (type.isa<FloatType>()) {
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
      if (type.isa<FloatType>()) {
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

struct AffineVectorReduceOpConversion
    : public OpConversionPattern<pxa::AffineVectorReduceOp> {
  using OpConversionPattern<pxa::AffineVectorReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pxa::AffineVectorReduceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto source = rewriter.create<AffineVectorLoadOp>(
        op.getLoc(), op.getVectorType(), op.mem(), op.idxs());
    // Get an attribute form of the map
    auto mapAttr = AffineMapAttr::get(op.map());
    // Set the map attribute
    source.setAttr(AffineVectorLoadOp::getMapAttrName(), mapAttr);
    auto reduce = createVectorReduction(rewriter, op, source.getResult());
    auto dest = rewriter.create<AffineVectorStoreOp>(
        op.getLoc(), ArrayRef<Type>{}, reduce, op.mem(), op.idxs());
    // Set the map attribute
    dest.setAttr(AffineVectorLoadOp::getMapAttrName(), mapAttr);
    op.replaceAllUsesWith(op.mem());
    rewriter.eraseOp(op);
    return mlir::success();
  }

  Value createVectorReduction(ConversionPatternRewriter &rewriter,
                              pxa::AffineVectorReduceOp op,
                              Value source) const {
    auto vectorType = op.getVectorType();
    switch (op.agg()) {
    case AggregationKind::assign:
      return op.vector();
    case AggregationKind::add: {
      if (vectorType.getElementType().isa<FloatType>()) {
        return rewriter.create<mlir::AddFOp>(op.getLoc(), source, op.vector());
      }
      return rewriter.create<mlir::AddIOp>(op.getLoc(), source, op.vector());
    }
    case AggregationKind::max: {
      if (vectorType.getElementType().isa<FloatType>()) {
        auto cmp = rewriter.create<mlir::CmpFOp>(
            op.getLoc(), mlir::CmpFPredicate::OGT, op.vector(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.vector(),
                                               source);
      }
      // TODO: determine whether to use signed or unsigned compare
      auto cmp = rewriter.create<mlir::CmpIOp>(
          op.getLoc(), mlir::CmpIPredicate::sgt, op.vector(), source);
      return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.vector(),
                                             source);
    }
    case AggregationKind::min: {
      if (vectorType.getElementType().isa<FloatType>()) {
        auto cmp = rewriter.create<mlir::CmpFOp>(
            op.getLoc(), mlir::CmpFPredicate::OLT, op.vector(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.vector(),
                                               source);
      }
      // TODO: determine whether to use signed or unsigned compare
      auto cmp = rewriter.create<mlir::CmpIOp>(
          op.getLoc(), mlir::CmpIPredicate::slt, op.vector(), source);
      return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.vector(),
                                             source);
    }
    case AggregationKind::mul: {
      if (vectorType.getElementType().isa<FloatType>()) {
        return rewriter.create<mlir::MulFOp>(op.getLoc(), source, op.vector());
      }
      return rewriter.create<mlir::MulIOp>(op.getLoc(), source, op.vector());
    }
    default:
      llvm_unreachable("Unsupported aggregation for "
                       "AffineVectorReduceOpConversion::createVectorReduction");
    }
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();
    if (op.isExternal()) {
      return mlir::success();
    }
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

struct LowerPXAToAffinePass
    : public LowerPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    PXAToAffineConversionTarget target(ctx);

    mlir::OwningRewritePatternList patterns;
    populatePXAToAffineConversionPatterns(patterns, &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().dump();
      emitError(mlir::UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

} // namespace

PXAToAffineConversionTarget::PXAToAffineConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<mlir::AffineDialect>();
  addLegalDialect<mlir::StandardOpsDialect>();
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

void populatePXAToAffineConversionPatterns(
    mlir::OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<                    //
      AffineParallelOpConversion,     //
      AffineIfOpConversion,           //
      AffineReduceOpConversion,       //
      AffineVectorReduceOpConversion, //
      FuncOpConversion,               //
      ReturnOpConversion>(ctx);
}

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

} // namespace pmlc::conversion::pxa_to_affine
