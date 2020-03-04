// Copyright 2019, Intel Corporation

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/Optional.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/tile/ir/ops.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::pxa {

enum MulOperationType {
  NoneMulOpType,
  FloatTy,
  IntTy,
};

struct GemmOperationMatch {
  MulOperationType mulOpType;
  mlir::AffineLoadOp in1Op;
  mlir::AffineLoadOp in2Op;
  AffineReduceOp outOp;
};

class Stencil {
private:
  // Number of instructions that need to be presented in the ParallelOp region
  // to be Considered a Gemm operation. For now they are affine.load,
  // affined.load, mul(f/i), reduce add, terminator.
  const unsigned kNumValidInstrInGemmRegion = 5;

  mlir::AffineParallelOp op;

public:
  explicit Stencil(mlir::AffineParallelOp opIn) : op(opIn) {}

  // Main function
  void DoStenciling();

  // Is this a Gemm operation
  mlir::Optional<GemmOperationMatch> getGemmOperation();
};

mlir::Optional<GemmOperationMatch> Stencil::getGemmOperation() {
  mlir::Optional<GemmOperationMatch> ret;
  Builder builder(op.getContext());
  auto *body = op.getBody();
  // Get the instructions in the body and match for load, load, mulXXX, reduce
  // add operations. For everything else we fail.
  if (body->getOperations().size() != kNumValidInstrInGemmRegion) {
    IVLOG(3, "the ParallelOp region didn't have the right number of "
             "instructions for a Gemm");
    return ret;
  }

  op.walk([&](AffineReduceOp reduceOp) {
    IVLOG(3, "Found ReduceOp");

    // Not check the reduceOp aggregation.
    if (reduceOp.agg() != AggregationKind::add) {
      IVLOG(3, "the reduce operation is not addition");
      return;
    }

    // Get the in tensors for the reduce op.
    Value reduceIn = reduceOp.val();
    MulOperationType mulOpType = MulOperationType::NoneMulOpType;

    // Make sure the in for the reduce is a result of a multiplication.
    auto valDef = reduceIn.getDefiningOp();

    if (!valDef) {
      IVLOG(3,
            "the source of the reduce operation is not defined in this block");
      return;
    }

    mlir::MulFOp mulfOp = llvm::dyn_cast_or_null<mlir::MulFOp>(valDef);
    mlir::MulIOp muliOp = llvm::dyn_cast_or_null<mlir::MulIOp>(valDef);
    if (!mulfOp && !muliOp) {
      IVLOG(3, "The source of the reduce is not a multiplication operation");
      return;
    }

    mlir::AffineLoadOp lhs;
    mlir::AffineLoadOp rhs;
    if (mulfOp) {
      mulOpType = MulOperationType::FloatTy;
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          mulfOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          mulfOp.rhs().getDefiningOp());
    } else if (muliOp) {
      mulOpType = MulOperationType::IntTy;
      lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          muliOp.lhs().getDefiningOp());
      rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
          muliOp.rhs().getDefiningOp());
    }

    // Now verify the types of the operands of the mulOp must be affine.load
    // operations.
    if (!lhs || !rhs) {
      IVLOG(3, "the lhs or rhs of the mul operation are not an affne.load "
               "operations");
      return;
    }

    // TODO: Need a bit better liveness analysis here to make sure the
    // parameters of any of the above 4 operations are not used in operations
    // with side effects - store, calls, etc.

    // Fill the values for the in/out/type of multiplication, etc.
    ret = GemmOperationMatch{mulOpType, lhs, rhs, reduceOp};
    op.setAttr("is_gemm", builder.getUnitAttr());
  });

  return ret;
}

void Stencil::DoStenciling() {
  // Initialization
  auto matchOpt = getGemmOperation();
  if (!matchOpt.hasValue()) {
    IVLOG(3, "Not a Gemm match.");
    return;
  }
}

void StencilPass::runOnFunction() {
  auto func = getFunction();
  func.walk([&](mlir::AffineParallelOp op) {
    Stencil as(op);
    as.DoStenciling();
  });
}

std::unique_ptr<mlir::Pass> createStencilPass() {
  return std::make_unique<StencilPass>();
}

} // namespace pmlc::dialect::pxa
