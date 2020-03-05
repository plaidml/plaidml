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

using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument, 8>;

// Number of tensors for the matrix multiplication
const unsigned kNumTensors = 3;
// Number of indices to search for (e.g. M, N, K)
const unsigned kNumIndex = 3;

class Stencil {
private:
  // Number of instructions that need to be presented in the ParallelOp region
  // to be Considered a Gemm operation. For now they are affine.load,
  // affined.load, mul(f/i), reduce add, terminator.
  const unsigned kNumValidInstrInGemmRegion = 5;

  // The ParallelOp that is being stencelled.
  mlir::AffineParallelOp op;

  // Target tensors, the first two are load, the third is aggregate
  llvm::SmallVector<Value, kNumTensors> tensors;

  // Target tensors strides, the first two are load, the third is aggregate
  llvm::SmallVector<mlir::StrideInfo, kNumTensors> tensorsStrides;

  // Set of the op's BlockArguments.
  BlockArgumentSet opBlockArguments;

  // Index in tensors
  BlockArgumentSet tensorIdxs[kNumTensors];
  // Stride one index for the tensors
  BlockArgumentSet strideOne[kNumTensors];
  // The indices used by the output tensor
  BlockArgumentSet outIdxs;
  // The accumulation indices
  BlockArgumentSet accIdxs;
  // All used indices
  BlockArgumentSet allIdxs;

  // Found Gemm operation data
  MulOperationType mulOpType;
  mlir::AffineLoadOp in1Op;
  mlir::AffineLoadOp in2Op;
  AffineReduceOp outOp;

  void PopulateOpBlockArgumentSet();
  BlockArgumentSet UsedIdxs(unsigned strideInfoIndex);
  void CollectUsedIndices();
  void CollectStrideOneIndices();
  void strideOneIdxs(unsigned indx);

public:
  explicit Stencil(mlir::AffineParallelOp opIn) : op(opIn) {}

  // Main function
  void DoStenciling();

  // Returns if a Gemm operation is identified.
  bool TryIdentifyGemmOperation();

  // Collect the tensors in the block
  bool CollectTensors();

  // Collect the StrideInfo of the tensors in the block
  bool ComputeStrideInfo();
};

bool Stencil::TryIdentifyGemmOperation() {
  auto *body = op.getBody();
  // Get the instructions in the body and match for load, load, mulXXX, reduce
  // add operations. For everything else we fail.
  if (body->getOperations().size() != kNumValidInstrInGemmRegion) {
    IVLOG(3, "the ParallelOp region didn't have the right number of "
             "instructions for a Gemm");
    return false;
  }

  auto beforeLastInstr = std::prev(body->end(), 2);
  AffineReduceOp reduceOp = llvm::dyn_cast<AffineReduceOp>(*beforeLastInstr);

  if (!reduceOp) {
    return false;
  }

  IVLOG(3, "Found ReduceOp");

  // Not check the reduceOp aggregation.
  if (reduceOp.agg() != AggregationKind::add) {
    IVLOG(3, "the reduce operation is not addition");
    return false;
  }

  // Get the in tensors for the reduce op.
  Value reduceIn = reduceOp.val();
  MulOperationType mulOpType = MulOperationType::NoneMulOpType;

  // Make sure the in for the reduce is a result of a multiplication.
  auto valDef = reduceIn.getDefiningOp();

  if (!valDef) {
    IVLOG(3, "the source of the reduce operation is not defined in this block");
    return false;
  }

  mlir::MulFOp mulfOp = llvm::dyn_cast_or_null<mlir::MulFOp>(valDef);
  mlir::MulIOp muliOp = llvm::dyn_cast_or_null<mlir::MulIOp>(valDef);
  if (!mulfOp && !muliOp) {
    IVLOG(3, "The source of the reduce is not a multiplication operation");
    return false;
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
  if (!lhs || !rhs || mulOpType == NoneMulOpType) {
    IVLOG(
        3,
        "the lhs or rhs of the mul operation are not an affne.load operations "
        "or the type of the multiplication is not on floats or ints.");
    return false;
  }

  // Fill the values for the in/out/type of multiplication, etc.
  this->mulOpType = mulOpType;
  in1Op = lhs;
  in2Op = rhs;
  outOp = reduceOp;
  return true;
}

bool Stencil::CollectTensors() {
  assert(in1Op && in2Op && outOp);

  tensors.push_back(in1Op.getMemRef());
  tensors.push_back(in2Op.getMemRef());
  tensors.push_back(outOp.out());

  return tensors.size() == kNumTensors;
}

bool Stencil::ComputeStrideInfo() {
  assert(in1Op && in2Op && outOp);

  auto in1OpOptional = computeStrideInfo(in1Op);
  auto in2OpOptional = computeStrideInfo(in2Op);
  auto outOpOptional = computeStrideInfo(outOp);

  if (!in1OpOptional || !in2OpOptional || !outOpOptional)
    return false;

  tensorsStrides.push_back(*in1OpOptional);
  tensorsStrides.push_back(*in2OpOptional);
  tensorsStrides.push_back(*outOpOptional);
  return tensorsStrides.size() == kNumTensors;
}

// Collect the non constant indices used to index the memref at specific index.
// Ignore indices that are constant for the ParallelOp.
BlockArgumentSet Stencil::UsedIdxs(unsigned strideInfoIndex) {
  assert(strideInfoIndex < kNumTensors);

  BlockArgumentSet used_idxs;
  for (auto kv : tensorsStrides[strideInfoIndex].strides) {
    // Make sure the BlockArgument is in the list of the ParallelOp's
    // BlockArguments.
    if (opBlockArguments.find(kv.first) != opBlockArguments.end()) {
      used_idxs.insert(kv.first);
    }
  }

  return used_idxs;
}

void Stencil::PopulateOpBlockArgumentSet() {
  for (auto blkArg : op.getBody()->getArguments()) {
    opBlockArguments.insert(blkArg);
  }
}

// Collect the indices that are not constants for the ParallelOp
// and also the accumulation indices.
void Stencil::CollectUsedIndices() {
  // The last tensor is the output.
  tensorIdxs[kNumIndex - 1] = UsedIdxs(kNumIndex - 1);
  outIdxs = tensorIdxs[kNumIndex - 1];
  accIdxs.clear();
  for (unsigned i = 0; i < kNumIndex - 1; ++i) {
    tensorIdxs[i] = UsedIdxs(i);
    for (auto idx : tensorIdxs[i]) {
      if (outIdxs.find(idx) == outIdxs.end()) {
        accIdxs.insert(idx);
      }
    }
  }

  // Add the out used indices to the all the used indices collection as well.
  allIdxs = accIdxs;
  allIdxs.insert(outIdxs.begin(), outIdxs.end());
}

// Get the indices with stride of one.
void Stencil::strideOneIdxs(unsigned indx) {
  for (auto kv : tensorsStrides[indx].strides) {
    if (kv.second == 1) {
      strideOne[indx].insert(kv.first);
    }
  }
}

void Stencil::CollectStrideOneIndices() {
  // Collect stride-one index
  for (unsigned i = 0; i < kNumTensors; ++i) {
    strideOneIdxs(i);
  }
}

void Stencil::DoStenciling() {
  // Initialization
  if (!TryIdentifyGemmOperation()) {
    IVLOG(3, "Not a Gemm match.");
    return;
  }

  if (!CollectTensors())
    return;

  if (!ComputeStrideInfo())
    return;

  PopulateOpBlockArgumentSet();
  CollectUsedIndices();
  CollectStrideOneIndices();

  op.setAttr("is_gemm", mlir::UnitAttr::get(op.getContext()));
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
