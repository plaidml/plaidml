// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/autostencil_pass.h"

#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "mlir/IR/Matchers.h"
#include "mlir/Translation.h"

#include "base/util/env.h"
#include "base/util/logging.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/transforms.h"
#include "pmlc/dialect/stripe/util.h"
#include "tile/targets/cpu/heatmap.h"

namespace pmlc {
namespace dialect {
namespace stripe {

using vertexai::tile::codegen::proto::MLIR_AutoStencilPass;
using vertexai::tile::targets::cpu::kHeatmapKeys;
using vertexai::tile::targets::cpu::kHeatmapSize;
using vertexai::tile::targets::cpu::kHeatmapValues;
using BlockArgumentSet = llvm::SmallPtrSet<mlir::BlockArgument, 8>;

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
  // The throughput and startup cost of M*N*K matrix multiplication
  std::pair<double, unsigned> Throughput(unsigned m, unsigned n, unsigned k);
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
  BlockArgumentSet RefUsedIdxs(Value value, bool with_conflict);
  // Test if idx is conflict with any index in innerIdxs
  bool ConflictInnerIndex(BlockArgument idx);
  // Test if idx in tensors[tensor_idx] is stride one index
  bool IsStrideOne(BlockArgument idx, unsigned tensor_idx);
  // Test if idx in the tensors are stride one
  bool ValidateStrideOne(BlockArgument idx, unsigned matrix_idx);
  // Test if idx exists in tensorIdxs[tensor_idx]
  bool IndexExists(BlockArgument idx, unsigned tensor_idx);
  // Test if idx exists in the right place
  bool ValidateIndexExistance(BlockArgument idx, unsigned matrix_idx);

  // Optimization options
  const MLIR_AutoStencilPass& options;
  // Stencil efficiency heatmap
  std::map<std::tuple<unsigned, unsigned, unsigned>, double> kHeatmap;
  // The current op
  ParallelForOp curOp;
  // Tensors' order
  unsigned tensorsOrder[kNumTensors];
  // M, N, K in inner block
  BlockArgument innerIdxs[kNumIndex];
  // M, N, K's tiles
  unsigned tiles[kNumIndex];
  // The matrix_idx for the next search
  unsigned nextMatrixIdx[kNumIndex] = {2, 3, 1};
  // Target tensors, the first two are load, the third is aggregate
  llvm::SmallVector<Value, kNumTensors> tensors;
  // Stride one index for the tensors
  BlockArgumentSet strideOne[kNumTensors];
  // The index used by the output tensor
  BlockArgumentSet outIdxs;
  // The accumulation index
  BlockArgumentSet accIdxs;
  // All used index
  BlockArgumentSet allIdxs;
  // Index in tensors
  BlockArgumentSet tensorIdxs[kNumTensors];
  // Each index has a set of conflict index that can't be choosed into inner block together
  llvm::DenseMap<BlockArgument, BlockArgumentSet> conflict;
  // The best performance
  double bestPerf;
  // The best tensors' order
  unsigned bestTensorsOrder[kNumTensors];
  // The best index (M, N, K)
  BlockArgument bestIdxs[kNumIndex];
  // The best tiles for (M, N, K)
  unsigned bestTiles[kNumIndex];
};

AutoStencil::AutoStencil(const MLIR_AutoStencilPass& opts) : options(opts) {
  for (unsigned i = 0; i < kHeatmapSize; ++i) {
    kHeatmap.emplace(std::make_tuple(kHeatmapKeys[i][0], kHeatmapKeys[i][1], kHeatmapKeys[i][2]), kHeatmapValues[i]);
  }
}

std::pair<double, unsigned> AutoStencil::Throughput(unsigned m, unsigned n, unsigned k) {
  auto iter = kHeatmap.find(std::make_tuple(m, n, k));
  if (iter != kHeatmap.end()) {
    return std::make_pair(iter->second, options.startup_cost());
  }
  // We mainly care about M and K. If both (m, n - 1, k) and (m, n + 1, k) exist,
  // we may use their average value for prediction
  auto iter0 = kHeatmap.find(std::make_tuple(m, n - 1, k));
  if (n == 1 || iter0 != kHeatmap.end()) {
    auto iter1 = kHeatmap.find(std::make_tuple(m, n + 1, k));
    if (iter1 != kHeatmap.end()) {
      return std::make_pair((n > 1) ? ((iter0->second + iter1->second) / 2) : iter1->second, options.startup_cost());
    }
  }
  // If we cannot find (m, n, k) in the heatmap, try the special cases
  for (const auto& spec : options.special_stencils()) {
    bool match = true;
    for (const auto& rule : spec.idxs()) {
      if (rule.name() == "m") {
        if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != m) {
          match = false;
          break;
        }
      } else if (rule.name() == "n") {
        if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != n) {
          match = false;
          break;
        }
      } else if (rule.name() == "k") {
        if (rule.size() > 0 && static_cast<unsigned>(rule.size()) != k) {
          match = false;
          break;
        }
      }
    }
    if (match) {
      return std::make_pair(0.001, spec.startup_cost());
    }
  }
  return std::make_pair(0.0, 0);
}

double AutoStencil::Evaluate() {
  unsigned tot_inner_loop = tiles[0] * tiles[1] * tiles[2];
  double throughput;
  unsigned startup_cost;
  std::tie(throughput, startup_cost) = Throughput(tiles[0], tiles[1], tiles[2]);
  if (throughput == 0) {
    return std::numeric_limits<double>::max();
  }
  double inner_time = tot_inner_loop / throughput;
  IVLOG(3, "Inner: loop = " << tot_inner_loop << " inner_time = " << inner_time);
  for (unsigned i = 0; i < kNumIndex; ++i) {
    IVLOG(3, idxName(innerIdxs[i]).str() << ": " << tiles[i]);
  }

  llvm::DenseMap<BlockArgument, unsigned> middle_idxs;
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

  llvm::DenseMap<BlockArgument, unsigned> outer_idxs;
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

  unsigned outer_batches = (tot_outer_loop - 1) / std::thread::hardware_concurrency() + 1;
  double perf = outer_batches * tot_middle_loop * (startup_cost + inner_time);
  IVLOG(3, "Performance = " << perf);

  return perf;
}

void AutoStencil::SearchTiles(unsigned idx) {
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
  if (options.only_po2()[idx]) {
    unsigned i = 1 << static_cast<unsigned>(std::floor(std::log2(range)));
    while (i > 0) {
      tiles[idx] = i;
      SearchTiles(idx + 1);
      i /= 2;
    }
  } else {
    for (unsigned i = range; i > 0; --i) {
      if (options.only_even()[idx] && (range % i != 0)) {
        continue;
      }
      tiles[idx] = i;
      SearchTiles(idx + 1);
    }
  }
}

bool AutoStencil::ConflictInnerIndex(BlockArgument idx) {
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
bool AutoStencil::IsStrideOne(BlockArgument idx, unsigned tensor_idx) {
  return strideOne[tensor_idx].find(idx) != strideOne[tensor_idx].end();
}

// Test if idx in the tensors are stride one
bool AutoStencil::ValidateStrideOne(BlockArgument idx, unsigned matrix_idx) {
  switch (matrix_idx) {
    case 0: {
      // Test if M is stride one for B(1) and C(2)
      return IsStrideOne(idx, tensorsOrder[1]) && IsStrideOne(idx, tensorsOrder[2]);
    }
    case 1: {
      // N is not restricted for stride one
      return true;
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

bool AutoStencil::IndexExists(BlockArgument idx, unsigned tensor_idx) {
  return tensorIdxs[tensor_idx].find(idx) != tensorIdxs[tensor_idx].end();
}

// Confirm if idx exists in the right place
bool AutoStencil::ValidateIndexExistance(BlockArgument idx, unsigned matrix_idx) {
  switch (matrix_idx) {
    case 0: {
      // Test if M exists in B and C, does not exist in A
      return !IndexExists(idx, tensorsOrder[0]) &&  //
             IndexExists(idx, tensorsOrder[1]) &&   //
             IndexExists(idx, tensorsOrder[2]);
    }
    case 1: {
      // Test if N exists in A and C, does not exist in B
      return IndexExists(idx, tensorsOrder[0]) &&   //
             !IndexExists(idx, tensorsOrder[1]) &&  //
             IndexExists(idx, tensorsOrder[2]);
    }
    case 2: {
      // Test if K exists in A and B, does not exist in C
      return IndexExists(idx, tensorsOrder[0]) &&  //
             IndexExists(idx, tensorsOrder[1]) &&  //
             !IndexExists(idx, tensorsOrder[2]);
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
    if (!ConflictInnerIndex(idx) &&            //
        ValidateStrideOne(idx, matrix_idx) &&  //
        ValidateIndexExistance(idx, matrix_idx)) {
      innerIdxs[matrix_idx] = idx;
      SearchIndex(nextMatrixIdx[matrix_idx]);
    }
  }
}

void AutoStencil::SearchTensorsOrder() {
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

// Collect the index used by value's RefineOp
// If with_conflict is true, specify the conflict between index, i.e., both index can't be in the inner block
BlockArgumentSet AutoStencil::RefUsedIdxs(Value value, bool with_conflict) {
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
  if (bestPerf == std::numeric_limits<double>::max()) {
    IVLOG(1, "No tile plan for stencil.");
    return;
  }

  llvm::SmallVector<std::pair<StringRef, unsigned>, 8> idxs;
  getAllIndex(curOp, &idxs);
  llvm::StringMap<unsigned> bestTileByName;
  for (unsigned i = 0; i < kNumIndex; ++i) {
    bestTileByName[idxName(bestIdxs[i])] = bestTiles[i];
  }
  llvm::SmallVector<int64_t, 8> inner_sizes;
  for (const auto& idx : idxs) {
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
  obody->walk([&](LoadOp op) {  //
    tensors.push_back(op.from());
  });
  if (tensors.size() != kNumTensors - 1) {
    return false;
  }
  // Collect all store ops to summarize the output tensors
  obody->walk([&](AggregateOp op) {  //
    tensors.push_back(op.into());
  });
  return tensors.size() == kNumTensors;
}

void AutoStencil::Stencil(ParallelForOp op) {
  // Initialization
  tensors.clear();
  bestPerf = std::numeric_limits<double>::max();
  curOp = op;

  if (!CollectTensors()) {
    return;
  }

  // The last tensor is the output.
  tensorIdxs[kNumIndex - 1] = RefUsedIdxs(tensors[kNumIndex - 1], true);
  outIdxs = tensorIdxs[kNumIndex - 1];
  accIdxs.clear();
  for (unsigned i = 0; i < kNumIndex - 1; ++i) {
    tensorIdxs[i] = RefUsedIdxs(tensors[i], true);
    for (auto idx : tensorIdxs[i]) {
      if (outIdxs.find(idx) == outIdxs.end()) {
        accIdxs.insert(idx);
      }
    }
  }
  allIdxs = accIdxs;
  allIdxs.insert(outIdxs.begin(), outIdxs.end());

  // Collect stride-one index
  for (unsigned i = 0; i < kNumTensors; ++i) {
    llvm::SmallVector<mlir::BlockArgument, 8> idxs;
    strideOneIdxs(tensors[i], &idxs);
    strideOne[i].insert(idxs.begin(), idxs.end());
  }

  // Search tensors' order, inner index and their tiles
  SearchTensorsOrder();

  // Transform
  Transform();
}

void AutoStencilPass::runOnFunction() {
#if defined(_WIN32) || defined(_WIN64)
  // As XSMM is not stable on Windows now, we disable this pass on Windows.
  // When the XSMM issue is solved, remove this section.
  return;
#endif

  auto reqs = vertexai::tile::stripe::FromProto(options.reqs());
  mlir::FuncOp f = getFunction();
  if (options.only_po2().size() != kNumIndex || options.only_even().size() != kNumIndex) {
    throw std::runtime_error("The size of only_po2 array or only_even array is incorrect.");
  }
  AutoStencil as(options);
  f.walk([&reqs, &as](ParallelForOp op) {
    if (hasAttrs(op.getOperation(), reqs)) {
      as.Stencil(op);
    }
  });
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
