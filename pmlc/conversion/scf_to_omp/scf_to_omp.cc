// Copyright 2020, Intel Corporation

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/scf_to_omp/pass_detail.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::scf_to_omp {

namespace scf = mlir::scf;
namespace omp = mlir::omp;

namespace {

struct ParallelOpConversion : public OpConversionPattern<scf::ParallelOp> {
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    IVLOG(2, "scf::ParallelOpConversion::matchAndRewrite");

    // Look for a parallel loop with the CPU thread tag
    // We expect that it will have a single index, whose range is the number
    // of threads which should execute it.
    // We will generate an OpenMP parallel block which will execute the
    // contents of this loop.
    // For the index variable, we will substitute omp_get_thread_num().
    // We expect that the body of the loop we are transforming will contain
    // some kind of appropriately tiled affine parallel structure which will
    // derive the index ranges from the thread number.

    bool isThread = hasUnitTag(op, cpuBlockTag());
    if (!isThread) {
      // we only care about parallel ops marked with the thread tag
      return success();
    }

    if (op.getNumLoops() != 1) {
      IVLOG(2, "Can't lower scf::ParallelOp unless number of indexes == 1");
      op.emitError("Number of loops must be 1");
      return failure();
    }

    auto loc = op.getLoc();

    // look up the loop bounds to get the number of threads to run
    Value lb = op.lowerBound().front();
    Value ub = op.upperBound().front();
    IVLOG(2, "DUMP OF LOOP BOUNDS:");
    lb.dump();
    ub.dump();

    // dummy thread count until we compute upper bounds value
    Value numThreads = rewriter.create<ConstantIndexOp>(loc, 8);

    // create the omp parallel fork/join region
    llvm::ArrayRef<mlir::Type> argTy;
    Attribute defaultValue, procBindValue;
    auto parallelOp = rewriter.create<omp::ParallelOp>(
        loc, argTy, Value(), numThreads,
        defaultValue.dyn_cast_or_null<StringAttr>(), ValueRange(), ValueRange(),
        ValueRange(), ValueRange(),
        procBindValue.dyn_cast_or_null<StringAttr>());

    // create a terminator (representing the join operation)
    rewriter.createBlock(&parallelOp.getRegion());
    auto &block = parallelOp.getRegion().back();
    rewriter.setInsertionPointToStart(&block);
    rewriter.create<omp::TerminatorOp>(loc);
    rewriter.setInsertionPointToStart(&block);

    // copy the body of the scf loop into the parallel region
    auto &insts = op.getBody()->getOperations();
    block.getOperations().splice(Block::iterator(parallelOp), insts,
                                 insts.begin(), std::prev(insts.end()));

    // generate a call to omp_get_thread_num()
    // ...or generate a dummy value, for the moment
    Value tid = rewriter.create<ConstantIndexOp>(loc, 42);

    // replace all references to the IV with the thread ID
    auto iv = op.getInductionVars().front();
    iv.replaceAllUsesWith(tid);

    return success();
  }
};

/// A pass converting SCF parallel loop into the OpenMP dialect.
struct LowerSCFToOpenMPPass
    : public LowerSCFToOpenMPBase<LowerSCFToOpenMPPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    ModuleOp module = getOperation();
    auto context = module.getContext();

    OwningRewritePatternList patterns;
    populateSCFToOpenMPConversionPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addLegalDialect<omp::OpenMPDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateSCFToOpenMPConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<mlir::Pass> createLowerSCFToOpenMPPass() {
  return std::make_unique<LowerSCFToOpenMPPass>();
}

} // namespace pmlc::conversion::scf_to_omp
