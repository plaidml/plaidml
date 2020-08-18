// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/cache.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {

using llvm::SmallVector;

struct SubgroupPlan {
  size_t subgroup_size;
  SmallVector<int64_t, 6> inner_tile;
  SmallVector<int64_t, 6> subgroup_tile;
  BlockArgument primary_subgroup;
};

AffineParallelOp tileAccumulations(AffineParallelOp op) {
  // Find the originating reduce
  assert(op.getNumResults() == 1);
  auto srcDef = getOriginalDef(op.getResult(0));
  auto red = mlir::cast<PxaReduceOp>(srcDef);
  // Get strides for output
  auto si = *computeStrideInfo(red);
  // Find all the accumulation indexes (stride 0 with respect to output) and
  // tile them into an inner block
  auto ranges = *op.getConstantRanges();
  SmallVector<int64_t, 6> accumTile;
  auto steps = op.steps().cast<ArrayAttr>().getValue();
  for (size_t i = 0; i < ranges.size(); i++) {
    auto arg = op.getIVs()[i];
    if (si.strides.count(arg)) {
      accumTile.push_back(steps[i].cast<IntegerAttr>().getInt());
    } else {
      accumTile.push_back(ranges[i]);
    }
  }
  return performTiling(op, accumTile);
}

void SubgroupApply(AffineParallelOp op, SubgroupPlan plan) {
  // Perform the primary innermost tiling
  auto inner = performTiling(op, plan.inner_tile);
  // Perform the deep inner tiling
  auto subgroup = performTiling(inner, plan.subgroup_tile);
  // Tile over accumulations
  auto accum = tileAccumulations(op);
  // Cache innermost loads at accum level
  subgroup.walk([&](PxaLoadOp load) { cacheLoad(accum, load); });
  // Cache innermost reduces at op level
  subgroup.walk([&](PxaReduceOp red) { cacheReduce(op, red); });
  // Vectorize everything we can
  op.walk(
      [&](AffineParallelOp par) { simpleVectorize(par, plan.subgroup_size); });
  // Try to 'vector cache' any remaining innermost loads
  subgroup.walk([&](PxaLoadOp load) { cacheLoadAsVector(inner, load); });
  // Convert local allocations to vector types
  op.walk([&](AllocOp alloc) { vectorizeBuffer(alloc); });
}

struct SubgroupsPass : public SubgroupsBase<SubgroupsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { doSubgroups(op); });
  }

  void doSubgroups(AffineParallelOp op) {
    if (op.getIVs().size() != 6) {
      return;
    }
    SubgroupPlan plan;
    plan.subgroup_size = 8;
    plan.inner_tile = {1, 8, 16, 1, 1, 8};
    plan.subgroup_tile = {1, 1, 8, 1, 1, 8};

    SubgroupApply(op, plan);
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

// Overly literal attempt to port cost model
#if 0
struct SubgroupOptions {
  // Subgroup sizes to consider
  SmallVector<int64_t, 4> subgroup_sizes = {8, 16};
  // Maximum register memory for a single subgroup (for each subgroup size)
  SmallVector<int64_t, 4> max_mem = {2240, 4480};
  // Latency to global memeory
  int64_t mem_latency = 420;
  // Latency to L2 memory
  int64_t l2_cache_latency = 125;
  // L2 Cache width (in bytes)
  int64_t cache_width = 64;
  // L2 Cache size
  int64_t cache_size = 3 * 768 * 1024;
  // The threshold of computations/memory_accesses to be memory bound
  double mem_bounded_threshold = 14;
  // Limit of inner block operations during unrolling
  int64_t inner_stmts_limit = 1250;
};

class SubgroupCostModel {
 public:
  //SubgroupCostModel(stripe::Block* block, const SubgroupPlan& plan, const
proto::SubgroupPass& options)
  //              : block_(block), plan_(plan), options_(options) {}



  // For a given operation and a given tile size, compute the total amount of
  // memory accesses excluding caching effects.  Basically, if the stride of a
  // given index is 0, we dont consider that index, otherwise the number of
  // accesses is just tht product of each tile size.
  size_t num_accesses(const Tiling& tile, Operation* op) const {
    const StrideInfo& si = op_strides.find(op)->second;
    size_t num = 1;
    for (const auto& kvp : tile) {
      if (si.strides.count(kvp.first)) {
        num *= kvp.second;
      }
    }
    return num;
  }

  // Given the tiling, and the strides of the output tensor,
  size_t num_work_items(Operation* out_op) const {
    const StrideInfo& out_strides = op_strides.find(out_op)->second;
    size_t num = 1;
    for (const auto& kvp : ranges) {
      // For cases which are not stride 0 relative the output
      if (out_strides.strides.count(kvp.first)) {
        // Begin with the range of the index
        size_t prod = kvp.second;
        // If it's subgrouped and not the primary thread, reduce
        if (plan.subgroup_tile.count(kvp.first) && plan.thread_idx != kvp.first)
{ prod /= plan.subgroup_tile.find(kvp.first)->second;
        }
        // If it's extra-tiles, reduce
        if (plan.extra_tile.count(kvp.first)) {
          prod /= plan.extra_tile.find(kvp.first)->second;
        }
        // Accumulate in
        num *= prod;
      }
    }
    return num;
  }


  // The primary parallel op we are doing analysis on
  AffineParallelOp ap_op;
  //.Precomputed range values for each index
  DenseMap<BlockArgument, int64_t> ranges;
  // Precomputed stride into for each operation
  DenseMap<Operation*, StrideInfo> op_strides;
  // The subgroup plan being evaluated
  SubgroupPlan plan;
};
#endif

} // namespace pmlc::dialect::pxa
