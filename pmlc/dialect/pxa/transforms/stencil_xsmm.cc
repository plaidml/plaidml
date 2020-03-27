// Copyright 2020 Intel Corporation

// TODO: includes etc

#include "stencil_generic.h"

namespace pmlc::dialect::pxa {

namespace {
// Default / standard stenciling functions

bool default_capture_fn(const mlir::AffineParallelOp& op,
                        llvm::SmallVector<mlir::Op, 2>* load_ops,
                        llvm::SmallVector<mlir::Op, 1>* compute_ops,
                        llvm::SmallVector<mlir::Op, 1>* store_ops)
{
  // Closely follows TryIdentifyGemmOperation from the previous stenciling pass
  // TODO: Do I want to clear the ops lists when I return false?

  assert(load_ops.empty() && compute_ops.empty() && store_ops.empty() && "expected op lists to be empty at start of capture");

  const unsigned kNumValidInstrInGemmRegion = 5;
  auto *body = op.getBody();
  // Looking for load..load..mul..reduce..terminator
  if (body->getOperations().size() != kNumValidInstrInGemmRegion) {
    IVLOG(3, "the ParallelOp region didn't have the right number of "
             "instructions for a Gemm");
    return false;
  }

  auto it = std::prev(body->end(), 2);
  auto reduceOp = llvm::dyn_cast<AffineReduceOp>(*it);
  if (!reduceOp) {
    return false;
  }
  store_ops->push_back(reduceOp);
  IVLOG(3, "Found ReduceOp");

  // Now check the reduceOp aggregation.
  if (reduceOp.agg() != AggregationKind::add) {
    IVLOG(3, "the reduce operation is not addition");
    return false;
  }

  // Get the operand for the reduce op and make sure it is the result of a
  // multiplication.
  auto defOp = reduceOp.val().getDefiningOp();
  if (!defOp) {
    IVLOG(3, "the source of the reduce operation is not defined in this block");
    return false;
  }

  mlir::AffineLoadOp lhs;
  mlir::AffineLoadOp rhs;
  if (auto mulfOp = llvm::dyn_cast_or_null<mlir::MulFOp>(defOp)) {
    compute_ops->push_back(mulfOp);
    mulOpType = MulOperandType::FloatTy;
    lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
        mulfOp.lhs().getDefiningOp());
    rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
        mulfOp.rhs().getDefiningOp());
  } else if (auto muliOp = llvm::dyn_cast_or_null<mlir::MulIOp>(defOp)) {
    compute_ops->push_back(muliOp);
    mulOpType = MulOperandType::IntTy;
    lhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
        muliOp.lhs().getDefiningOp());
    rhs = llvm::dyn_cast_or_null<mlir::AffineLoadOp>(
        muliOp.rhs().getDefiningOp());
  } else {
    IVLOG(3, "The source of the reduce is not a multiplication operation");
    return false;
  }

  // Now verify the types of the operands of the mulOp must be affine.load
  // operations.
  if (!lhs || !rhs || mulOpType == MulOperandType::None) {
    IVLOG(3,
          "the lhs or rhs of the mul operation are not affine.load operations "
          "or the type of the multiplication is not on floats or ints.");
    return false;
  }
  load_ops->push_back(lhs);
  load_ops->push_back(rhs);

  return true;
}

std::list<TensorAndIndexPermutation> default_preflight_fn(
    const llvm::SmallVector<mlir::Op, 2>& load_ops,
    const llvm::SmallVector<mlir::Op, 1>& compute_ops,
    const llvm::SmallVector<mlir::Op, 1>& store_ops)
{
  // TODO
}

// TODO: Might be able to do a prettier generator than "return a list"?
std::list<StencilTiling> default_tiling_generator(const TensorAndIndexPermutation& permutation)
{
  // TODO
}

double default_cost_fn(const StencilTiling& tiling,
                       const TensorAndIndexPermutation& permutation,
                       const llvm::SmallVector<mlir::Op, 2>& load_ops,
                       const llvm::SmallVector<mlir::Op, 1>& compute_ops,
                       const llvm::SmallVector<mlir::Op, 1>& store_ops)
{
  // TODO
}

}  // namespace

class StencilXSMM : public StencilGeneric {
private:
  // TODO

public:
  explicit StencilXSMM(mlir::AffineParallelOp op)
      : StencilGeneric{ 
          op,
          default_capture_fn,
          default_preflight_fn,
          default_tiling_generator,
          default_cost_fn
        }
  {
      // TODO ctor
  }

}

} // namespace pmlc::dialect::pxa
