// Copyright 2019, Intel Corporation

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "base/util/logging.h"
#include "pmlc/dialect/pxa/dialect.h"
#include "pmlc/dialect/pxa/ops.h"
#include "pmlc/dialect/pxa/passes.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::pxa {

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

struct AffineParallelForOpConversion : public LoweringBase<AffineParallelForOp> {
  explicit AffineParallelForOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(AffineParallelForOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
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

struct ReduceOpConversion : public LoweringBase<ReduceOp> {
  explicit ReduceOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(ReduceOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    auto in = rewriter.create<AffineLoadOp>(op.getLoc(), op.out(), op.map(), op.idxs());
    Value* agg = nullptr;
    switch (op.agg()) {
      case AggregationKind::add:
        agg = rewriter.create<mlir::AddFOp>(op.getLoc(), in, op.val());
        break;
      case AggregationKind::mul:
        agg = rewriter.create<mlir::MulFOp>(op.getLoc(), in, op.val());
        break;
      default:
        llvm_unreachable("Unsupported aggregation for CreateInit");
    }
    rewriter.create<AffineStoreOp>(op.getLoc(), agg, op.out(), op.map(), op.idxs());
    rewriter.create<AffineTerminatorOp>(op.getLoc());
    rewriter.eraseOp(op);
  }
};

void LoweringPass::runOnModule() {
  // Set up target (i.e. what is legal)
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineOpsDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addIllegalDialect<Dialect>();

  // Setup rewrite patterns
  mlir::OwningRewritePatternList patterns;
  patterns.insert<AffineParallelForOpConversion>(&getContext());
  patterns.insert<ReduceOpConversion>(&getContext());

  // Run the conversion
  if (failed(applyPartialConversion(getModule(), target, patterns, nullptr))) {
    getModule().dump();
    emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> pxa\n");
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<mlir::Pass> createLowerToAffinePass() {  //
  return std::make_unique<LoweringPass>();
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "pxa-legalize-to-affine",                               //
    "Legalize from PXA dialect to Affine dialect");

}  // namespace pmlc::dialect::pxa
