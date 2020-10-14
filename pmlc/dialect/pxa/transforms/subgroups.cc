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
#include "pmlc/dialect/pxa/transforms/tile_accumulate.h"
#include "pmlc/dialect/pxa/transforms/vectorize.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"
#include "pmlc/util/util.h"

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
    // Tile accumulations only works on parallel loops with a single result
    if (op.getNumResults() != 1) {
      return;
    }
    // Skip low dimensional cases.  It's not profitable to subgroup such cases.
    // However, additionally running the subgroup transform anyway should be
    // safe, but actually causes backend test resnetDense to fail
    // TODO: Debug correctness failure of resnetDense when this code is removed.
    if (op.getIVs().size() < 3) {
      return;
    }
    // Verify we have constant ranges and cache
    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      return;
    }
    ranges = *maybeRanges;
    IVLOG(3, "Ranges = " << *maybeRanges);
    // Preflight all loads/stores + cache
    bool safe = true;
    op.walk([&](PxaReduceOp red) {
      if (red.agg() != AtomicRMWKind::addf) {
        // This isn't really unsafe, but basically this test removes
        // non-contraction like ops from consideration.  Eltwise ops are not
        // good to subgroup due to low computational density (we should
        // explicity check for that), and pooling fail due to vectorization
        // issues with CMP (we should fix that).  However, this works for now.
        safe = false;
        return;
      }
      if (!preflightIO(red)) {
        safe = false;
      }
    });
    op.walk([&](PxaLoadOp load) {
      if (!preflightIO(load)) {
        safe = false;
      }
    });
    if (!safe) {
      return;
    }
    plan.innerTile.resize(ranges.size());
    plan.subgroupTile.resize(ranges.size());
    for (int64_t subgroupSize : params.subgroupSizes) {
      plan.subgroupSize = subgroupSize;
      computeCostRecursive(0);
    }
  }

  template <typename OpType>
  bool preflightIO(OpType ioOp) {
    IVLOG(3, "Preflight: " << debugString(*ioOp.getOperation()));
    if (ioOp.getOperation()->getBlock() != op.getBody()) {
      IVLOG(3, "Not part of block");
      return false;
    }
    auto maybeFlatStrides = computeStrideInfo(ioOp);
    if (!maybeFlatStrides) {
      IVLOG(3, "Not strided");
      return false;
    }
    flat_strides.push_back(*maybeFlatStrides);
    IVLOG(3, "  flat strides = " << *maybeFlatStrides);

    auto maybeDimStrides =
        computeStrideInfo(ioOp.getAffineMap(), ioOp.getMapOperands());
    if (!maybeDimStrides) {
      IVLOG(3, "Not strided");
      return false;
    }
    dim_strides.push_back(*maybeDimStrides);
    IVLOG(3, "  dimensional strides = " << *maybeDimStrides)

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

  double memory_io(double cache_elems,
                   SmallVector<int64_t, 4> tensor_dimensions,
                   SmallVector<int64_t, 4> tensor_strides) {
    // Start with one cache line
    double cache_lines = 1.0;
    // Current accumulated maximum value
    int64_t max_val = 0;
    // For each dimension (in sorted order)
    assert(tensor_dimensions.size() == tensor_strides.size());
    for (size_t i = tensor_dimensions.size(); i > 0; i--) {
      // Compute gap per step
      int64_t gap = std::abs(tensor_strides[i - 1]) - max_val;
      // Multiply current cache hits by size
      cache_lines *= static_cast<double>(tensor_dimensions[i - 1]);
      // Compute probability that cache line is shared across gap
      double prob_shared = 0.0; // Assume it's never shared
      if (cache_elems != 0.0 && gap < cache_elems) {
        prob_shared = 1.0 - (gap / cache_elems);
      }
      // Subtract shared elements
      cache_lines -=
          prob_shared * static_cast<double>(tensor_dimensions[i - 1] - 1);
      // Update max_val
      max_val +=
          std::abs(tensor_strides[i - 1]) * (tensor_dimensions[i - 1] - 1);
    }
    return cache_lines;
  }

  struct MemInfo {
    unsigned subgroupCount;
    int64_t registers;
    int64_t accesses;
    double cacheMiss;
    SmallVector<int64_t, 4> tensor_dimensions;
    SmallVector<int64_t, 4> tensor_strides;
  };

  MemInfo computeMemoryInfo(SmallVectorImpl<StrideInfo> &dim_strides,
                            StrideInfo flat_strides) {
    MemInfo out = {0, 1, 1, 0.0};
    for (auto si : dim_strides) {
      int64_t pos = 0;
      // TODO: add neg

      bool has_tensor_stride = false;
      int64_t tensor_stride = -1;
      for (auto kvp : si.strides) {
        auto index = kvp.first.getArgNumber();
        auto dim_stride = kvp.second;

        if (flat_strides.strides.count(kvp.first)) {
          auto flat_stride = flat_strides.strides[kvp.first];

          if (!has_tensor_stride) {
            has_tensor_stride = true;
            tensor_stride = flat_stride / dim_stride;
          }
          assert(tensor_stride == flat_stride / dim_stride);

          if (tensor_stride == 1 &&
              plan.subgroupTile[index] == plan.subgroupSize) {
            out.subgroupCount++;
          }
        }

        pos += dim_stride * (plan.innerTile[index] - 1);

        out.accesses *= plan.innerTile[index];
      }
      if (tensor_stride != -1) {
        out.registers *= pos + 1;
        out.tensor_dimensions.push_back(pos + 1);
        out.tensor_strides.push_back(tensor_stride);
      }
    }

    double kCacheWidth = 64.0;
    // TODO: fixed sizeof
    out.cacheMiss =
        memory_io(kCacheWidth / 4, out.tensor_dimensions, out.tensor_strides);
    return out;
  }

  double computeCost() {
    int64_t totRegisters = 0;
    int64_t totAccesses = 0;
    double totCacheMiss = 0.0;

    for (size_t i = 0; i < flat_strides.size(); i++) {
      auto mi = computeMemoryInfo(dim_strides[i], flat_strides[i]);
      if (i == 0 && mi.subgroupCount == 0) {
        return std::numeric_limits<double>::infinity();
      }

      if (mi.subgroupCount) {
        mi.registers /= plan.subgroupSize;
        mi.accesses /= plan.subgroupSize;
        mi.cacheMiss /= plan.subgroupSize;
      }

      totRegisters += mi.registers;
      totAccesses += mi.accesses;
      totCacheMiss += mi.cacheMiss;
    }

    if (totRegisters > params.maxRegsPerThread) {
      IVLOG(3, "Invalid subgroup plan: " << plan << ", totRegisters = "
                                         << totRegisters);
      return std::numeric_limits<double>::infinity();
    }
    IVLOG(3, "Valid subgroup plan: " << plan
                                     << ", totRegisters = " << totRegisters);

    int64_t totOps = 1;
    for (auto it : plan.innerTile) {
      totOps *= it;
    }

    double kCacheLatency = 125.0;
    double kMemoryLatency = 420.0;
    double totMemIO = (totAccesses - totCacheMiss) * kCacheLatency +
                      totCacheMiss * kMemoryLatency;

    return totMemIO / totOps;
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
  SmallVector<StrideInfo, 4> flat_strides;
  SmallVector<SmallVector<StrideInfo, 4>, 4> dim_strides;
};

void SubgroupApply(AffineParallelOp op, SubgroupPlan plan) {
  // Perform the primary innermost tiling
  auto inner = performTiling(op, plan.innerTile);
  // Perform the deep inner tiling
  auto subgroup = performTiling(inner, plan.subgroupTile);
  // Tile over accumulations
  auto accum = tileAccumulations(op, false);
  // Cache innermost loads at accum level
  subgroup.walk([&](PxaLoadOp load) { cacheLoad(accum, load); });
  // Cache innermost reduces at op level
  subgroup.walk([&](PxaReduceOp reduce) { cacheReduce(op, reduce); });
  // Vectorize everything we can
  op.walk([&](AffineParallelOp par) {
    vectorizeOverOutputs(par, plan.subgroupSize);
  });
  // Try to 'vector cache' any remaining innermost loads
  subgroup.walk([&](PxaLoadOp load) {
    cacheLoadAsVector(inner, load, plan.subgroupSize);
  });
  // Convert local allocations to vector types
  op.walk([&](AllocOp alloc) { vectorizeBuffer(alloc); });
  // Attach subgroup size
  setIntegerTag(op, subgroupSizeTag(), plan.subgroupSize);
}

struct SubgroupsPass : public SubgroupsBase<SubgroupsPass> {
  void runOnFunction() final {
    auto func = getFunction();
    func.walk([&](AffineParallelOp op) { doSubgroups(op); });
  }

  void doSubgroups(AffineParallelOp op) {
    SubgroupParams params = {
        {8, 16}, // Subgroup sizes to consider
        40,      // Maximum register per thread
    };
    SubgroupCostModel cm(params, op);
    if (cm.bestCost == std::numeric_limits<double>::infinity()) {
      // If subgrouping fails, we tile accumulations instead to handle the other
      // cases
      tileAccumulations(op, false);
      setIntegerTag(op, subgroupSizeTag(), 1);
      return;
    }
    IVLOG(3, "best plan = " << cm.bestPlan);
    SubgroupApply(op, cm.bestPlan);
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

} // namespace pmlc::dialect::pxa
