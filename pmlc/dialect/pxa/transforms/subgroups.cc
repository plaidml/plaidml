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
  int64_t subgroupSize;
  SmallVector<int64_t, 6> innerTile;
  SmallVector<int64_t, 6> subgroupTile;
};

std::ostream &operator<<(std::ostream &os, const SubgroupPlan &plan) {
  size_t sz = plan.innerTile.size();
  os << plan.subgroupSize << ":[";
  for (size_t i = 0; i < sz; i++) {
    os << plan.innerTile[i];
    if (i != sz - 1) {
      os << " ";
    }
  }
  os << "]:[";
  for (size_t i = 0; i < sz; i++) {
    os << plan.subgroupTile[i];
    if (i != sz - 1) {
      os << " ";
    }
  }
  os << "]";
  return os;
}

struct SubgroupParams {
  SmallVector<int64_t, 4> subgroupSizes;
  int64_t maxRegsPerThread;
};

struct SubgroupCostModel {
  SubgroupCostModel(const SubgroupParams &params, AffineParallelOp op)
      : params(params), op(op) {
    bestCost = std::numeric_limits<double>::infinity();
    // Verify that we have only 1 results
    if (op.getNumResults() != 1) {
      return;
    }
    // Verify we have constant ranges and cache
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      return;
    }
    ranges = *maybeRanges;
    // Preflight all loads/stores + cache
    bool safe = true;
    op.walk([&](PxaLoadOp load) {
      if (!preflightIO(load)) {
        safe = false;
      }
    });
    op.walk([&](PxaReduceOp red) {
      if (!preflightIO(red)) {
        safe = false;
      }
    });
    plan.innerTile.resize(ranges.size());
    plan.subgroupTile.resize(ranges.size());
    for (int64_t subgroupSize : params.subgroupSizes) {
      plan.subgroupSize = subgroupSize;
      computeCostRecursive(0);
    }
  }

  template <typename OpType>
  bool preflightIO(OpType ioOp) {
    if (ioOp.getOperation()->getBlock() != op.getBody()) {
      return false;
    }
    auto maybeStrides = computeStrideInfo(ioOp);
    if (!maybeStrides) {
      return false;
    }
    ioStrides.push_back(*maybeStrides);
    return true;
  }

  void computeCostRecursive(unsigned idx) {
    if (idx == ranges.size()) {
      auto cost = computeCost();
      if (cost < bestCost) {
        bestCost = cost;
        bestPlan = plan;
      }
      return;
    }
    // Try using a subgroup or not for each index
    for (unsigned doSubgroup = 0; doSubgroup < 2; doSubgroup++) {
      // Get the range of this index
      int64_t range = ranges[idx];
      if (range % plan.subgroupSize != 0 && doSubgroup == 1) {
        continue; // Skip the subgroup if not even divison by subgroup size
      }
      if (doSubgroup) {
        // If we are doing subgrouping, set + reduce range
        plan.subgroupTile[idx] = plan.subgroupSize;
        range /= plan.subgroupSize;
      } else {
        plan.subgroupTile[idx] = 1;
      }
      // Now try all even divisors for remaining size
      for (int64_t ts = 1; ts <= range; ts++) {
        if (range % ts != 0) {
          continue;
        }
        plan.innerTile[idx] = ts * plan.subgroupTile[idx];
        computeCostRecursive(idx + 1);
      }
    }
    return;
  }

  struct MemInfo {
    unsigned subgroupCount;
    int64_t memSize;
  };

  // Compute the memory info for a single load/store given the current plan
  MemInfo computeMemoryInfo(StrideInfo si) {
    MemInfo out = {0, 1};
    for (unsigned i = 0; i < ranges.size(); i++) {
      auto iv = op.getIVs()[i];
      if (si.strides.count(iv)) {
        if (plan.subgroupTile[i] == plan.subgroupSize && si.strides[iv] == 1) {
          out.subgroupCount++;
        }
        out.memSize *= plan.innerTile[i];
      }
    }
    if (out.subgroupCount) {
      out.memSize /= plan.subgroupSize;
    }
    return out;
  }

  double computeCost() {
    // Compute memory usage
    int64_t totMemory = 0;
    for (auto si : ioStrides) {
      auto mi = computeMemoryInfo(si);
      if (mi.subgroupCount > 1) {
        return std::numeric_limits<double>::infinity();
      }
      totMemory += mi.memSize;
    }
    if (totMemory / plan.subgroupSize > params.maxRegsPerThread) {
      return std::numeric_limits<double>::infinity();
    }
    IVLOG(3, "Valid subgroup plan: " << plan << ", totMemory = " << totMemory);
    int64_t groups = 1;
    for (size_t i = 0; i < ranges.size(); i++) {
      groups *= ranges[i] / plan.innerTile[i];
    }
    return static_cast<double>(groups * plan.subgroupSize);
  }

  // The parameters to the cost model
  SubgroupParams params;
  // The operation being optimized
  AffineParallelOp op;
  // Current plan
  SubgroupPlan plan;
  // Best cost found so far
  double bestCost;
  // Best plan found for far
  SubgroupPlan bestPlan;
  // Cache of the index ranges
  SmallVector<int64_t, 8> ranges;
  // Strides for all io ops
  SmallVector<StrideInfo, 4> ioStrides;
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
  for (unsigned i = 0; i < ranges.size(); i++) {
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
  auto inner = performTiling(op, plan.innerTile);
  // Perform the deep inner tiling
  auto subgroup = performTiling(inner, plan.subgroupTile);
  // Tile over accumulations
  auto accum = tileAccumulations(op);
  // Cache innermost loads at accum level
  subgroup.walk([&](PxaLoadOp load) { cacheLoad(accum, load); });
  // Cache innermost reduces at op level
  subgroup.walk([&](PxaReduceOp red) { cacheReduce(op, red); });
  // Vectorize everything we can
  op.walk(
      [&](AffineParallelOp par) { simpleVectorize(par, plan.subgroupSize); });
  // Try to 'vector cache' any remaining innermost loads
  subgroup.walk([&](PxaLoadOp load) {
    cacheLoadAsVector(inner, load, plan.subgroupSize);
  });
  // Convert local allocations to vector types
  op.walk([&](AllocOp alloc) { vectorizeBuffer(alloc); });
}

struct SubgroupsPass : public SubgroupsBase<SubgroupsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { doSubgroups(op); });
  }

  void doSubgroups(AffineParallelOp op) {
    SubgroupParams params = {
        {8, 16}, // Subgroup sizes
        6,       // Maximum register per thread
    };
    SubgroupCostModel cm(params, op);
    if (cm.bestCost == std::numeric_limits<double>::infinity()) {
      return;
    }
    IVLOG(1, "best plan = " << cm.bestPlan);
    SubgroupApply(op, cm.bestPlan);
    // SubgroupApply(op, plan);
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

} // namespace pmlc::dialect::pxa
