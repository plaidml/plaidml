// Copyright 2020 Intel Corporation

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

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
  // Tensors' order
  unsigned tensorsOrder[kNumTensors];
  // M, N, K in inner block
  mlir::BlockArgument innerIdxs[kNumIndex];
  // The matrix_idx for the next search
  unsigned nextMatrixIdx[kNumIndex] = {2, 3, 1};
  // M, N, K's tiles
  unsigned tiles[kNumIndex];

  // Even index split only. Specified for each index.
  bool onlyEven[kNumIndex] = {true, true, true};

  // Found Gemm operation data
  MulOperationType mulOpType;
  mlir::AffineLoadOp in1Op;
  mlir::AffineLoadOp in2Op;
  AffineReduceOp outOp;
  // The best tensors' order
  unsigned bestTensorsOrder[kNumTensors];
  // The best index (M, N, K)
  mlir::BlockArgument bestIdxs[kNumIndex];
  // The best tiles for (M, N, K)
  unsigned bestTiles[kNumIndex];

  // The best performance
  double bestPerf;

  llvm::SmallVector<int64_t, 8> opConstRanges;

  void PopulateOpBlockArgumentSet();
  BlockArgumentSet UsedIdxs(unsigned strideInfoIndex);
  void CollectUsedIndices();
  void CollectStrideOneIndices();
  void strideOneIdxs(unsigned indx);

  // Search tensors' order
  void SearchTensorsOrder();
  // Search the index, i.e., M, N, K, for the inner block
  void SearchIndex(unsigned matrix_idx);
  // Test if idx in tensors[tensor_idx] is stride one index
  bool IsStrideOne(mlir::BlockArgument idx, unsigned tensor_idx);
  // Test if idx in the tensors are stride one
  bool ValidateStrideOne(mlir::BlockArgument idx, unsigned matrix_idx);
  // Test if idx exists in tensorIdxs[tensor_idx]
  bool IndexExists(mlir::BlockArgument idx, unsigned tensor_idx);
  // Test if idx exists in the right place
  bool ValidateIndexExistance(mlir::BlockArgument idx, unsigned matrix_idx);
  // For (M, N, K) in the inner block, search their tiles
  void SearchTiles(unsigned idx);

  int64_t idxRange(mlir::BlockArgument idx);

  // Evaluate the performance of the current searching state
  double Evaluate();

  // Set the number of threads to a default value if not set from outside.
  void SetNumberThreads();

  // Number of threads
  unsigned numThreads;

  // The throughput and startup cost of M*N*K matrix multiplication
  StencilCostFunction costFn;

public:
  explicit Stencil(mlir::AffineParallelOp op, int numThreads,
                   StencilCostFunction costFn)
      : op(op), numThreads(numThreads), costFn(costFn) {
    assert(numThreads && "numThreads must be non-zero!");
  }

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

// Test if idx in tensors[tensor_idx] is stride one index
bool Stencil::IsStrideOne(mlir::BlockArgument idx, unsigned tensor_idx) {
  return strideOne[tensor_idx].find(idx) != strideOne[tensor_idx].end();
}

// Test if idx in the tensors are stride one
bool Stencil::ValidateStrideOne(mlir::BlockArgument idx, unsigned matrix_idx) {
  switch (matrix_idx) {
  case 0:
    return IsStrideOne(idx, tensorsOrder[1]) &&
           IsStrideOne(idx, tensorsOrder[2]);
  case 1:
    return true;
  case 2:
    return IsStrideOne(idx, tensorsOrder[0]);
  default:
    llvm_unreachable("Wrong matrix_idx");
  }
  return false;
}

bool Stencil::IndexExists(mlir::BlockArgument idx, unsigned tensor_idx) {
  return tensorIdxs[tensor_idx].find(idx) != tensorIdxs[tensor_idx].end();
}

// Confirm if idx exists in the right place
bool Stencil::ValidateIndexExistance(mlir::BlockArgument idx,
                                     unsigned matrix_idx) {
  switch (matrix_idx) {
  case 0:
    // Test if M exists in B and C, does not exist in A
    return !IndexExists(idx, tensorsOrder[0]) && //
           IndexExists(idx, tensorsOrder[1]) &&  //
           IndexExists(idx, tensorsOrder[2]);
  case 1:
    // Test if N exists in A and C, does not exist in B
    return IndexExists(idx, tensorsOrder[0]) &&  //
           !IndexExists(idx, tensorsOrder[1]) && //
           IndexExists(idx, tensorsOrder[2]);
  case 2:
    // Test if K exists in A and B, does not exist in C
    return IndexExists(idx, tensorsOrder[0]) && //
           IndexExists(idx, tensorsOrder[1]) && //
           !IndexExists(idx, tensorsOrder[2]);
  default:
    llvm_unreachable("Wrong matrix_idx.");
  }
  return false;
}

// Search for matrix index (0 for M, 1 for N, 2 for K)
void Stencil::SearchIndex(unsigned matrix_idx) {
  if (matrix_idx >= kNumIndex) {
    // We have the index and then search the tiles for these index
    SearchTiles(0);
    return;
  }
  auto &idxs = (matrix_idx == kNumIndex - 1) ? allIdxs : outIdxs;
  for (auto idx : idxs) {
    if (ValidateStrideOne(idx, matrix_idx) &&
        ValidateIndexExistance(idx, matrix_idx)) {
      innerIdxs[matrix_idx] = idx;
      SearchIndex(nextMatrixIdx[matrix_idx]);
    }
  }
}

void Stencil::SearchTensorsOrder() {
  // A B C, Search M(0) first as M is most restricted index
  tensorsOrder[0] = 0;
  tensorsOrder[1] = 1;
  tensorsOrder[2] = 2;
  SearchIndex(0);
  // B A C, Search M(0) first as M is most restricted index
  tensorsOrder[0] = 1;
  tensorsOrder[1] = 0;
  SearchIndex(0);
}

void Stencil::SearchTiles(unsigned idx) {
  if (idx >= kNumIndex) {
    double performance = Evaluate();
    if (performance < bestPerf) {
      bestPerf = performance;
      for (unsigned i = 0; i < kNumTensors; ++i) {
        bestTensorsOrder[i] = tensorsOrder[i];
      }
      for (unsigned i = 0; i < kNumIndex; ++i) {
        bestIdxs[i] = innerIdxs[i];
        bestTiles[i] = tiles[i];
      }
    }
    return;
  }

  unsigned range = idxRange(innerIdxs[idx]);
  for (unsigned i = range; i > 0; --i) {
    if (onlyEven[idx] && (range % i != 0)) {
      continue;
    }
    tiles[idx] = i;
    SearchTiles(idx + 1);
  }
}

int64_t Stencil::idxRange(mlir::BlockArgument idx) {
  assert(idx.getArgNumber() < opConstRanges.size());
  return opConstRanges[idx.getArgNumber()];
}

double Stencil::Evaluate() {
  unsigned tot_inner_loop = tiles[0] * tiles[1] * tiles[2];
  // tiles 0, 1, 2 --> m, n, k
  auto cost = costFn(tiles);
  if (cost.throughput == 0) {
    return std::numeric_limits<double>::max();
  }
  double inner_time = tot_inner_loop / cost.throughput;
  IVLOG(3,
        "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
  for (unsigned i = 0; i < kNumIndex; ++i) {
    IVLOG(3, innerIdxs[i] << ": " << tiles[i]);
  }

  llvm::DenseMap<mlir::BlockArgument, unsigned> middle_idxs;
  for (auto idx : accIdxs) {
    middle_idxs.try_emplace(idx, idxRange(idx));
  }
  for (unsigned i = 0; i < kNumIndex; ++i) {
    auto it = middle_idxs.find(innerIdxs[i]);
    if (it != middle_idxs.end()) {
      it->second = (it->second - 1) / tiles[i] + 1;
    }
  }
  unsigned tot_middle_loop = 1;
  for (auto &kvp : middle_idxs) {
    tot_middle_loop *= kvp.second;
  }

  IVLOG(3, "Middle: loop = " << tot_middle_loop);

  for (auto &kvp : middle_idxs) {
    if (kvp.second > 1) {
      IVLOG(3, kvp.first << ": " << kvp.second);
    }
  }

  llvm::DenseMap<mlir::BlockArgument, unsigned> outer_idxs;
  for (auto idx : outIdxs) {
    outer_idxs.try_emplace(idx, idxRange(idx));
  }
  for (unsigned i = 0; i < kNumIndex; ++i) {
    auto it = outer_idxs.find(innerIdxs[i]);
    if (it != outer_idxs.end()) {
      it->second = (it->second - 1) / tiles[i] + 1;
    }
  }
  unsigned tot_outer_loop = 1;
  for (auto &kvp : outer_idxs) {
    tot_outer_loop *= kvp.second;
  }

  IVLOG(3, "Outer: loop = " << tot_outer_loop);

  for (auto &kvp : outer_idxs) {
    if (kvp.second > 1) {
      IVLOG(3, kvp.first << ": " << kvp.second);
    }
  }

  unsigned outer_batches = (tot_outer_loop - 1) / numThreads + 1;
  double perf =
      outer_batches * tot_middle_loop * (cost.startupCost + inner_time);

  IVLOG(3, "Performance = " << perf);
  return perf;
}

void Stencil::DoStenciling() {
  // Initialization
  tensors.clear();
  bestPerf = std::numeric_limits<double>::max();
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

  auto ranges = op.getConstantRanges();
  if (!ranges)
    return;

  opConstRanges = *ranges;

  // Search tensors' order, inner index and their tiles
  SearchTensorsOrder();

  IVLOG(1, "Best Perf: " << bestPerf);
  IVLOG(1, "Best Tiles: " << bestTiles[0] << ":" << bestTiles[1] << ":"
                          << bestTiles[2]);

  if (bestPerf == std::numeric_limits<double>::max()) {
    IVLOG(1, "No tile plan for stencil.");
    return;
  }

  op.setAttr("is_gemm", mlir::UnitAttr::get(op.getContext()));
}

struct StencilPass : public mlir::FunctionPass<StencilPass> {
  StencilPass() { assert(false && "StencilPass must be configured"); }
  StencilPass(const StencilPass &) {}
  StencilPass(unsigned numThreads_, StencilCostFunction costFn)
      : costFn(costFn) {
    numThreads = numThreads_;
  }

  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](mlir::AffineParallelOp op) {
      Stencil stencil(op, numThreads.getValue(), costFn);
      stencil.DoStenciling();
    });
  }

  Option<unsigned> numThreads{
      *this, "threads",
      llvm::cl::desc("Specifies number of threads for the stencil pass")};

  StencilCostFunction costFn;
};

std::unique_ptr<mlir::Pass> createStencilPass(unsigned numThreads,
                                              StencilCostFunction costFn) {
  return std::make_unique<StencilPass>(numThreads, costFn);
}

} // namespace pmlc::dialect::pxa
