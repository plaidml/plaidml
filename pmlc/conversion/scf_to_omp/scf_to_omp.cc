// Copyright 2020, Intel Corporation

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
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

LogicalResult convertSCPParallel(scf::ParallelOp op) {
  IVLOG(2, "convertSCPParallel");
  OpBuilder builder(op);

  // Look for a parallel loop.  We expect that it will have a single index,
  // whose range is the number of threads which should execute it.  We will
  // generate an OpenMP parallel block which will execute the contents of this
  // loop.  For the index variable, we will substitute omp_get_thread_num().
  // We expect that the body of the loop we are transforming will contain some
  // kind of appropriately tiled affine parallel structure which will derive
  // the index ranges from the thread number.

  if (op.getNumLoops() != 1) {
    IVLOG(2, "Can't lower scf::ParallelOp unless number of indexes == 1");
    op.emitError("Number of loops must be 1");
    return failure();
  }

  auto loc = op.getLoc();

  // Look up the loop bounds to get the number of threads to run.
  APInt lowerAP, upperAP, stepAP;
  if (!matchPattern(op.lowerBound().front(), m_ConstantInt(&lowerAP)) ||
      !matchPattern(op.upperBound().front(), m_ConstantInt(&upperAP)) ||
      !matchPattern(op.step().front(), m_ConstantInt(&stepAP))) {
    IVLOG(2, "Unable to get constant bounds in scf::ParallelOp");
    op.emitError("scf.parallel bounds must be constant");
  }

  // Verify everthing is normalized.
  uint64_t lower = lowerAP.getLimitedValue();
  uint64_t upper = upperAP.getLimitedValue();
  uint64_t step = stepAP.getLimitedValue();
  IVLOG(2, "lower = " << lower << ", upper = " << upper << ", step = " << step);
  if (lower != 0 || step != 1) {
    op.emitError("Non-normalized loops not supported");
    return failure();
  }

  auto clauseAttr =
      StringAttr::get(stringifyClauseDefault(omp::ClauseDefault::defshared),
                      builder.getContext());
  // Create the omp parallel fork/join region
  auto parallelOp = builder.create<omp::ParallelOp>(
      loc,
      /*if_expr_va=*/Value(),                      //
      /*num_threads_var=*/op.upperBound().front(), //
      /*default_val=*/clauseAttr,                  //
      /*private_vars=*/ValueRange(),               //
      /*firstprivate_vars=*/ValueRange(),          //
      /*shared_vars=*/ValueRange(),                //
      /*copyin_vars=*/ValueRange(),                //
      /*proc_bind_val=*/StringAttr());

  // create a terminator (representing the join operation)
  builder.createBlock(&parallelOp.getRegion());
  auto *block = &parallelOp.getRegion().back();
  builder.setInsertionPointToStart(block);
  auto termOp = builder.create<omp::TerminatorOp>(loc);
  (void)termOp;

  // copy the body of the scf loop into the parallel region
  auto &insts = op.getBody()->getOperations();
  block->getOperations().splice(Block::iterator(termOp), insts, insts.begin(),
                                std::prev(insts.end()));

  // generate a call to omp_get_thread_num()
  // ...or generate a dummy value, for the moment
  builder.setInsertionPointToStart(block);
  Value tid = builder.create<ConstantIndexOp>(loc, 42);
  (void)tid;

  // replace all references to the IV with the thread ID
  auto iv = op.getInductionVars().front();
  iv.replaceAllUsesWith(tid);

  // Delete old loop + return
  op.erase();
  return success();
}

/// A pass converting SCF parallel loop into the OpenMP dialect.
struct LowerSCFToOpenMPPass
    : public LowerSCFToOpenMPBase<LowerSCFToOpenMPPass> {
  // Run the dialect converter on the module.
  void runOnOperation() final {
    Operation *op = getOperation();
    op->walk([&](scf::ParallelOp op) {
      if (failed(convertSCPParallel(op))) {
        signalPassFailure();
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerSCFToOpenMPPass() {
  return std::make_unique<LowerSCFToOpenMPPass>();
}

} // namespace pmlc::conversion::scf_to_omp
