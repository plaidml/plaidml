// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/autostencil_pass.h"

#include <sstream>
#include <vector>

#include "mlir/IR/Matchers.h"
#include "mlir/Translation.h"

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transforms.h"
#include "pmlc/dialect/stripe/util.h"

#include "base/util/logging.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using vertexai::tile::codegen::proto::MLIR_AutoStencilPass;
using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument*, kIndexLimit>;

// Number of tensors for the matrix multiplication
const unsigned kNumTensors = 3;
// Number of searching index, i.e., M, N, K
const unsigned kNumIndex = 3;

class AutoStencil {
public:
  explicit AutoStencil(const MLIR_AutoStencilPass& opts);
  // Main function
  void Stencil(ParallelForOp op);
private:
  // The throughput of M*N*K matrix multiplication
  double Throughput(unsigned m, unsigned n, unsigned k);
  // Evaluate the performance of the current searching state
  double Evaluate();
  // Search tensors' order
  void SearchTensorsOrder();
  // Search the index, i.e., M, N, K, for the inner block
  void SearchIndex(unsigned matrix_idx);
  // For (M, N, K) in the inner block, search their tiles
  void SearchTiles(unsigned idx);
  // Collect the tensors in the block
  bool CollectTensors();
  // Transform the current ParallelForOp
  void Transform();

  // Collect the index used by value's RefineOp
  BlockArgumentSet RefUsedIdxs(Value* value, bool with_conflict);
  // Test if idx is conflict with any index in innerIdxs
  bool ConflictInnerIndex(BlockArgument* idx);
  // Test if idx in tensors[tensor_idx] is stride one index
  bool IsStrideOne(BlockArgument* idx, unsigned tensor_idx);
  // Test if idx in the tensors are stride one
  bool ValidateStrideOne(BlockArgument* idx, unsigned matrix_idx);

  // Number of processors
  const MLIR_AutoStencilPass& options;
  // The current op
  ParallelForOp curOp;
  // Tensors' order
  unsigned tensorsOrder[kNumTensors];
  // M, N, K in inner block
  BlockArgument* innerIdxs[kNumIndex];
  // M, N, K's tiles
  unsigned tiles[kNumIndex];
  // The matrix_idx for the next search
  unsigned nextMatrixIdx[kNumIndex] = {3, 2, 0};
  // Target tensors, the first two are load, the third is aggregate
  llvm::SmallVector<Value*, kNumTensors> tensors;
  // Stride one index for the tensors
  BlockArgumentSet strideOne[kNumTensors];
  // The index used by the output tensor
  BlockArgumentSet outIdxs;
  // The accumulation index
  BlockArgumentSet accIdxs;
  // All used index
  BlockArgumentSet allIdxs;
  // Each index has a set of conflict index that can't be choosed into inner block together
  llvm::DenseMap<BlockArgument*, BlockArgumentSet> conflict;
  // The best performance
  double bestPerf;
  // The best tensors' order
  unsigned bestTensorsOrder[kNumTensors];
  // The best index (M, N, K)
  BlockArgument* bestIdxs[kNumIndex];
  // The best tiles for (M, N, K)
  unsigned bestTiles[kNumIndex];
};

AutoStencil::AutoStencil(const MLIR_AutoStencilPass& opts) :
  options(opts) {
}

double AutoStencil::Throughput(unsigned m, unsigned n, unsigned k) {
  if (m == 16 && n == 16 && k == 3) {
    return 64;
  }
  if (m == 32 && k == 32) {
    return 64;
  }
  if (m == 16 && k == 16) {
    return 64;
  }
  if (m == 8 && k == 16) {
    return 64;
  }
  if (m == 48 && k == 48) {
    return 64;
  }
  return -1;
}

double AutoStencil::Evaluate() {
  unsigned tot_inner_loop = tiles[0] * tiles[1] * tiles[1];
  double inner_time = tot_inner_loop / Throughput(tiles[0], tiles[1], tiles[2]);
  IVLOG(3, "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
  for (unsigned i = 0; i < kNumIndex; ++i) {
    IVLOG(3, idxName(innerIdxs[i]).str() << ": " << tiles[i]);
  }

  llvm::DenseMap<BlockArgument*, unsigned> middle_idxs;
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
  for (auto& kvp : middle_idxs) {
    tot_middle_loop *= kvp.second;
  }
  IVLOG(3, "Middle: loop = " << tot_middle_loop);
  for (auto& kvp : middle_idxs) {
    if (kvp.second > 1) {
      IVLOG(3, idxName(kvp.first).str() << ": " << kvp.second);
    }
  }

  llvm::DenseMap<BlockArgument*, unsigned> outer_idxs; 
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
  for (auto& kvp : outer_idxs) {
    tot_outer_loop *= kvp.second;
  }
  IVLOG(3, "Outer: loop = " << tot_outer_loop);
  for (auto& kvp : outer_idxs) {
    if (kvp.second > 1) {
      IVLOG(3, idxName(kvp.first).str() << ": " << kvp.second);
    }
  }

  unsigned outer_batches = (tot_outer_loop - 1) / options.processors() + 1;
  double perf = outer_batches * tot_middle_loop * inner_time;
  IVLOG(3, "Performance = " << perf);

  return perf;
}

void AutoStencil::SearchTiles(unsigned idx) {
  if (idx >= kNumIndex) {
    double performance = Evaluate();
    if (performance > bestPerf) {
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
  if (options.only_po2()[idx]) {
    unsigned i = 1;
    while (i <= range) {
      tiles[idx] = i;
      SearchTiles(idx + 1);
      i *= 2;
    }
  }
  else {
    for (unsigned i = 1; i <= range; ++i) {
      if (options.only_even()[idx] && (range % i != 0)) {
        continue;
      }
      tiles[idx] = i;
      SearchTiles(idx + 1);
    }
  }
}

bool AutoStencil::ConflictInnerIndex(BlockArgument* idx) {
  BlockArgumentSet& conflict_set = conflict[idx];
  for (auto& elem : innerIdxs) {
    if (elem == idx) {
      return true;
    }
    if (conflict_set.find(elem) != conflict_set.end()) {
      return true;
    }
  }
  return false;
}

// Test if idx in tensors[tensor_idx] is stride one index
bool AutoStencil::IsStrideOne(BlockArgument* idx, unsigned tensor_idx) {
  return strideOne[tensor_idx].find(idx) != strideOne[tensor_idx].end();
}

// Test if idx in the tensors are stride one
bool AutoStencil::ValidateStrideOne(BlockArgument* idx, unsigned matrix_idx) {
  switch (matrix_idx) {
    case 0: {
      // M is not restricted for stride one
      return true;
    }
    case 1: {
      // Test if N is stride one for B(1) and C(2)
      return IsStrideOne(idx, tensorsOrder[1]) && IsStrideOne(idx, tensorsOrder[2]);
    }
    case 2: {
      // Test if K is stride one for A(0)
      return IsStrideOne(idx, tensorsOrder[0]);
    }
    default: {
      throw std::runtime_error("Wrong matrix_idx.");
    }
  }
  return false;
}

// Search for matrix index (0 for M, 1 for N, 2 for K)
void AutoStencil::SearchIndex(unsigned matrix_idx) {
  if (matrix_idx >= kNumIndex) {
    // We have the index and then search the tiles for these index
    SearchTiles(0);
    return;
  }
  auto& idxs = (matrix_idx == kNumIndex - 1) ? allIdxs : outIdxs;
  for (auto idx : idxs) {
    if (!ConflictInnerIndex(idx) && ValidateStrideOne(idx, matrix_idx)) {
      innerIdxs[matrix_idx] = idx;
      SearchIndex(nextMatrixIdx[matrix_idx]);
    }
  }
}

void AutoStencil::SearchTensorsOrder() {
  // A B C, Search N(1) first as N is most restricted index
  tensorsOrder[0] = 0;
  tensorsOrder[1] = 1;
  tensorsOrder[2] = 2;
  SearchIndex(1);
  // B A C, Search N(1) first as N is most restricted index
  tensorsOrder[0] = 1;
  tensorsOrder[1] = 0;
  SearchIndex(1);
}

// Collect the index used by value's RefineOp
// If with_conflict is true, specify the conflict between index, i.e., both index can't be in the inner block
BlockArgumentSet AutoStencil::RefUsedIdxs(Value* value, bool with_conflict) {
  BlockArgumentSet used_idxs;
  auto ref_op = mlir::dyn_cast<RefineOp>(value->getDefiningOp());
  auto tensor = value->getType().cast<TensorRefType>();
  if (!ref_op) {
    throw std::runtime_error("Access a tensor not defined by RefineOp.");
  }
  for (int i = 0; i < tensor.getRank(); i++) {
    auto access = AffinePolynomial(ref_op.getOffset(i));
    for (auto [arg0, scale0] : access.terms) {
      used_idxs.insert(arg0);
      if (with_conflict) {
        for (auto [arg1, scale1] : access.terms) {
          if (arg0 != arg1) {
            conflict[arg0].insert(arg1);
            conflict[arg1].insert(arg0);
          }
        }
      }
    }
  }
  return used_idxs;
}

void AutoStencil::Transform() {
  if (bestPerf < 0) {
    IVLOG(1, "No tile plan for stencil.");
    return;
  }
  llvm::SmallVector<std::pair<StringRef, unsigned>, kIndexLimit> idxs = getAllIndex(curOp);
  llvm::StringMap<unsigned> bestTileByName;
  for (unsigned i = 0; i < kNumIndex; ++i) {
    bestTileByName[idxName(bestIdxs[i])] = bestTiles[i];
  }
  llvm::SmallVector<int64_t, kIndexLimit> inner_sizes;
  for (auto& idx : idxs) {
    auto it = bestTileByName.find(idx.first);
    inner_sizes.push_back(it == bestTileByName.end() ? 1 : it->second);
  }
  // The first tiling: split outer&middle and inner
  Tile(curOp, inner_sizes); 
  ParallelForOp inner = mlir::dyn_cast<ParallelForOp>(curOp.inner().front().front());

  // Set ParallelForOp tags
  setOpAttrUnit(curOp, curOp.getBodyBuilder(), "mac");
  setOpAttrUnit(inner, inner.getBodyBuilder(), "mac_inner");
  setOpAttrUnit(inner, inner.getBodyBuilder(), "xsmm");

  // Set RefineOp tags
  setOpAttrUnit(tensors[bestTensorsOrder[0]]->getDefiningOp(), inner.getBodyBuilder(), "A");
  setOpAttrUnit(tensors[bestTensorsOrder[1]]->getDefiningOp(), inner.getBodyBuilder(), "B");
  setOpAttrUnit(tensors[bestTensorsOrder[2]]->getDefiningOp(), inner.getBodyBuilder(), "C");

  // Set index tags
  StringRef m_idx = idxName(bestIdxs[0]);
  setIdxAttrUnit(inner, m_idx, "stencil");
  setIdxAttrUnit(inner, m_idx, "stencil_m");
  StringRef n_idx = idxName(bestIdxs[1]);
  setIdxAttrUnit(inner, n_idx, "stencil");
  setIdxAttrUnit(inner, n_idx, "stencil_n");
  StringRef k_idx = idxName(bestIdxs[2]);
  setIdxAttrUnit(inner, k_idx, "stencil");
  setIdxAttrUnit(inner, k_idx, "stencil_k");
}

bool AutoStencil::CollectTensors() {
  // Make a builder to write to just before the terminator
  Block* obody = curOp.getBody();

  // Collect all load ops to summarize the input tensors
  obody->walk([&](LoadOp op) {
    tensors.push_back(op.from());
  });
  if (tensors.size() != kNumTensors - 1) {
    return false;
  }
  // Collect all store ops to summarize the output tensors
  obody->walk([&](AggregateOp op) {
    tensors.push_back(op.into());
  });
  return tensors.size() == kNumTensors;
}

void AutoStencil::Stencil(ParallelForOp op) {
  // Initialization
  tensors.clear();
  bestPerf = -1;
  curOp = op;

  if (!CollectTensors()) {
    return;
  }

  // The last tensor is the output.
  outIdxs = RefUsedIdxs(tensors[kNumIndex - 1], true);
  accIdxs.clear();
  for (unsigned i = 0; i < kNumIndex - 1; ++i) {
    BlockArgumentSet used_idxs = RefUsedIdxs(tensors[i], true);
    for (auto idx : used_idxs) {
      if (outIdxs.find(idx) == outIdxs.end()) {
        accIdxs.insert(idx);
      }
    }
  }
  allIdxs = accIdxs;
  allIdxs.insert(outIdxs.begin(), outIdxs.end());

  // Collect stride-one index
  for (unsigned i = 0; i < kNumTensors; ++i) {
    auto idxs = strideOneIdxs(tensors[i]);
    strideOne[i].insert(idxs.begin(), idxs.end());
  }

  // Search tensors' order, inner index and their tiles
  SearchTensorsOrder();

  // Transform
  Transform();
}

void AutoStencilPass::runOnFunction() {
  auto reqs = vertexai::tile::stripe::FromProto(options.reqs());
  mlir::FuncOp f = getFunction();
  if (options.only_po2().size() != kNumIndex || options.only_even().size() != kNumIndex) {
    throw std::runtime_error("The size of only_po2 array or only_even array is incorrect.");
  }
  AutoStencil as(options);
  f.walk([&reqs, &as] (ParallelForOp op) {
    if (hasAttrs(op.getOperation(), reqs)) {
      as.Stencil(op);
    }
  });
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
