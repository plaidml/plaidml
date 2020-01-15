// Copyright 2020, Intel Corporation

#include "pmlc/conversion/pxa_to_affine/pxa_to_affine.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/pxa/ir/dialect.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::pxa_to_affine {

namespace pxa = dialect::pxa;

using mlir::AffineLoadOp;
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
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::PatternMatchResult;
using mlir::RankedTensorType;
using mlir::ReturnOp;
using mlir::Type;
using mlir::Value;

using mlir::edsc::AffineLoopNestBuilder;
using mlir::edsc::IndexHandle;
using mlir::edsc::ScopedContext;
using mlir::edsc::ValueHandle;
using mlir::edsc::intrinsics::constant_index;

using util::AggregationKind;

namespace {

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override;
};

template <typename OpType>
struct LoweringBase : public OpConversionPattern<OpType> {
  MLIRContext* ctx;

  explicit LoweringBase(MLIRContext* ctx) : OpConversionPattern<OpType>(ctx), ctx(ctx) {}
  PatternMatchResult match(Operation* op) const override { return this->matchSuccess(); }
};

struct AffineParallelForOpConversion : public LoweringBase<pxa::AffineParallelForOp> {
  explicit AffineParallelForOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(pxa::AffineParallelForOp op, ArrayRef<Value> operands,
               ConversionPatternRewriter& rewriter) const override {
    // Create an Affine loop nest following the order of ranges.
    ScopedContext scope(rewriter, op.getLoc());
    llvm::SmallVector<ValueHandle, 8> affineIvs;
    llvm::SmallVector<ValueHandle*, 8> affineIvPtrs;
    llvm::SmallVector<ValueHandle, 8> affineLbs;
    llvm::SmallVector<ValueHandle, 8> affineUbs;
    llvm::SmallVector<int64_t, 8> affineSteps;
    auto ranges = op.ranges().getValue();

    for (size_t i = 0; i < ranges.size(); i++) {
      affineIvs.emplace_back(IndexHandle());
      affineIvPtrs.emplace_back(&affineIvs.back());
      affineLbs.emplace_back(constant_index(0));
      affineUbs.emplace_back(constant_index(ranges[i].cast<IntegerAttr>().getInt()));
      affineSteps.emplace_back(1);
    }

    // Build the empty Affine loop nest with an innermost loop body containing a terminator.
    AffineLoopNestBuilder(affineIvPtrs, affineLbs, affineUbs, affineSteps)();

    // Replace all uses of old values
    size_t idx = 0;
    for (auto arg : op.inner().front().getArguments()) {
      arg->replaceAllUsesWith(affineIvs[idx++].getValue());
    }

    // Move ParallelForOp's operations (single block) to Affine innermost loop.
    auto& innermostLoopOps = mlir::getForInductionVarOwner(affineIvs[ranges.size() - 1]).getBody()->getOperations();
    auto& stripeBodyOps = op.inner().front().getOperations();
    innermostLoopOps.erase(innermostLoopOps.begin(), innermostLoopOps.end());
    innermostLoopOps.splice(innermostLoopOps.begin(), stripeBodyOps, stripeBodyOps.begin(), stripeBodyOps.end());

    // We are done. Remove original op.
    rewriter.eraseOp(op);
  }
};

struct AffineReduceOpConversion : public LoweringBase<pxa::AffineReduceOp> {
  explicit AffineReduceOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(pxa::AffineReduceOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override {
    auto source = rewriter.create<AffineLoadOp>(op.getLoc(), op.out(), op.map(), op.idxs());
    auto reduce = createReduction(rewriter, op, source.getResult());
    rewriter.create<AffineStoreOp>(op.getLoc(), reduce, op.out(), op.map(), op.idxs());
    rewriter.eraseOp(op);
  }

  Value createReduction(ConversionPatternRewriter& rewriter, pxa::AffineReduceOp op, Value source) const {
    switch (op.agg()) {
      case AggregationKind::assign:
        return source;
      case AggregationKind::add: {
        if (source.getType().isa<FloatType>()) {
          return rewriter.create<mlir::AddFOp>(op.getLoc(), source, op.val());
        }
        return rewriter.create<mlir::AddIOp>(op.getLoc(), source, op.val());
      }
      case AggregationKind::max: {
        if (source.getType().isa<FloatType>()) {
          auto cmp = rewriter.create<mlir::CmpFOp>(op.getLoc(), mlir::CmpFPredicate::OGT, op.val(), source);
          return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(), source);
        }
        // TODO: determine whether to use signed or unsigned compare
        auto cmp = rewriter.create<mlir::CmpIOp>(op.getLoc(), mlir::CmpIPredicate::sgt, op.val(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(), source);
      }
      case AggregationKind::min: {
        if (source.getType().isa<FloatType>()) {
          auto cmp = rewriter.create<mlir::CmpFOp>(op.getLoc(), mlir::CmpFPredicate::OLT, op.val(), source);
          return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(), source);
        }
        // TODO: determine whether to use signed or unsigned compare
        auto cmp = rewriter.create<mlir::CmpIOp>(op.getLoc(), mlir::CmpIPredicate::slt, op.val(), source);
        return rewriter.create<mlir::SelectOp>(op.getLoc(), cmp, op.val(), source);
      }
      case AggregationKind::mul: {
        if (source.getType().isa<FloatType>()) {
          return rewriter.create<mlir::MulFOp>(op.getLoc(), source, op.val());
        }
        return rewriter.create<mlir::MulIOp>(op.getLoc(), source, op.val());
      }
      default:
        llvm_unreachable("Unsupported aggregation for AffineReduceOpConversion::createReduction");
    }
  }
};

void LoweringPass::runOnModule() {
  // Set up target (i.e. what is legal)
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineOpsDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<pxa::Dialect>();

  // Setup rewrite patterns
  mlir::OwningRewritePatternList patterns;
  patterns.insert<AffineParallelForOpConversion>(&getContext());
  patterns.insert<AffineReduceOpConversion>(&getContext());

  // Run the conversion
  if (failed(applyPartialConversion(getModule(), target, patterns, nullptr))) {
    getModule().dump();
    emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> pxa\n");
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> createLowerPXAToAffinePass() {  //
  return std::make_unique<LoweringPass>();
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "convert-pxa-to-affine",                                //
    "Convert from PXA dialect to Affine dialect");

}  // namespace pmlc::conversion::pxa_to_affine
