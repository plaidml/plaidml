// Copyright 2020 Intel Corporation

#include "llvm/Support/Process.h"

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
  os << plan.subgroupSize << ",";
  for (size_t i = 0; i < sz; i++) {
    os << plan.innerTile[i];
    if (i != sz - 1) {
      os << ",";
    }
  }
  os << ",";
  for (size_t i = 0; i < sz; i++) {
    os << plan.subgroupTile[i];
    if (i != sz - 1) {
      os << ",";
    }
  }
  os << ",";
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
    IVLOG(1, "RANGES = " << *maybeRanges);
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
    IVLOG(1, "Preflight: " << debugString(*ioOp.getOperation()));
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
    IVLOG(1, "  flat strides = " << *maybeFlatStrides);

    auto maybeDimStrides =
        computeStrideInfo(ioOp.getAffineMap(), ioOp.getMapOperands());
    if (!maybeDimStrides) {
      IVLOG(3, "Not strided");
      return false;
    }
    dim_strides.push_back(*maybeDimStrides);
    IVLOG(1, "  dim strides = " << *maybeDimStrides)

    return true;
  }

  void computeCostRecursive(unsigned idx) {
    // force V0 best plan
    /*
    plan.innerTile[0] = 1;
    plan.innerTile[1] = 64;
    plan.innerTile[2] = 7;
    plan.innerTile[3] = 1;
    plan.innerTile[4] = 4;
    plan.innerTile[5] = 1;

    plan.subgroupTile[0] = 1;
    plan.subgroupTile[1] = 16;
    plan.subgroupTile[2] = 1;
    plan.subgroupTile[3] = 1;
    plan.subgroupTile[4] = 1;
    plan.subgroupTile[5] = 1;

    idx = ranges.size();
    */
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

  struct MemInfo2 {
    int64_t registers;
    int64_t accesses;
    SmallVector<int64_t, 4> dimensions;
    SmallVector<int64_t, 4> tensor_strides;
    double cacheMiss;
    unsigned subgroup;
  };

  double memory_io(size_t cache_width, SmallVector<int64_t, 4> dimensions,
                   SmallVector<int64_t, 4> strides) {
    // TODO: fixed sizeof
    double cache_elems = static_cast<double>(cache_width) / 4;

    // Start with one cache line
    double cache_lines = 1.0;

    // Current accumulated maximum value
    int64_t max_val = 0;

    // TODO: check that dimensiosn / strides are same size
    // For each dimension (in sorted order)
    for (size_t i = dimensions.size(); i > 0; i--) {
      // Compute gap per step
      int64_t gap = std::abs(strides[i - 1]) - max_val;

      // Multiply current cache hits by size
      cache_lines *= static_cast<double>(dimensions[i - 1]);

      // Compute probability that cache line is shared across gap
      double prob_shared = 0.0; // Assume it's never shared
      if (cache_elems != 0.0 && gap < cache_elems) {
        prob_shared = 1.0 - (gap / cache_elems);
      }

      // Subtract shared elements
      cache_lines -= prob_shared * static_cast<double>(dimensions[i - 1] - 1);

      // Update max_val
      max_val += std::abs(strides[i - 1]) * (dimensions[i - 1] - 1);
    }
    return cache_lines;
  }

  MemInfo2 computeMemoryInfo2(SmallVectorImpl<StrideInfo> &dim_strides,
                              StrideInfo flat_strides) {
    MemInfo2 out = {1, 1, {}, {}, 0.0, 0};
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
            out.subgroup++;
          }
        }

        pos += dim_stride * (plan.innerTile[index] - 1);

        out.accesses *= plan.innerTile[index];
      }
      if (tensor_stride != -1) {
        out.registers *= pos + 1;
        out.dimensions.push_back(pos + 1);
        out.tensor_strides.push_back(tensor_stride);
      }
    }

    out.cacheMiss = memory_io(64, out.dimensions, out.tensor_strides);
    return out;
  }

  int64_t num_work_items(StrideInfo output_strides) {
    int64_t work_items = 1;
    for (unsigned i = 0; i < ranges.size(); i++) {
      auto iv = op.getIVs()[i];
      if (output_strides.strides.count(iv)) {
        work_items *= ranges[i] / (plan.innerTile[i] / plan.subgroupTile[i]);
      }
    }
    return work_items;
  }

  double computeCost() {
    total_plans++;
    // full scan
    // if (diag.next() == util::DiagnosticCounter::Result::Match) {
    //   return -std::numeric_limits<double>::infinity();
    // }

    std::string tileinfo;
    int64_t totRegisters = 0;
    int64_t totAccesses = 0;
    double totCacheMiss = 0.0;

    for (size_t i = 0; i < flat_strides.size(); i++) {
      auto mi2 = computeMemoryInfo2(dim_strides[i], flat_strides[i]);
      if (i == 0 && mi2.subgroup == 0) {
        return std::numeric_limits<double>::infinity();
      }

      bool printx = false;
      for (auto dim : mi2.dimensions) {
        if (printx) {
          tileinfo += "x";
        } else {
          printx = true;
        }
        tileinfo += std::to_string(dim);
      }
      tileinfo += ",";

      tileinfo += std::to_string(mi2.registers);
      tileinfo += ",";
      tileinfo += std::to_string(mi2.accesses);
      tileinfo += ",";
      tileinfo += std::to_string(mi2.cacheMiss);
      tileinfo += ",";
      tileinfo += std::to_string(mi2.subgroup);
      tileinfo += ",";

      if (mi2.subgroup) {
        mi2.registers /= plan.subgroupSize;
        mi2.accesses /= plan.subgroupSize;
        mi2.cacheMiss /= plan.subgroupSize;
      }

      totRegisters += mi2.registers;
      totAccesses += mi2.accesses;
      totCacheMiss += mi2.cacheMiss;
    }

    if (totRegisters > params.maxRegsPerThread) {
      IVLOG(3, "Invalid subgroup plan: " << plan << ", totRegisters = "
                                         << totRegisters);
      return std::numeric_limits<double>::infinity();
    }
    IVLOG(3, "Valid subgroup plan: " << plan
                                     << ", totRegisters = " << totRegisters);

    int64_t groups = num_work_items(flat_strides[0]);
    double v1cost = static_cast<double>(groups * plan.subgroupSize);

    int64_t totOps = 1;
    for (auto it : plan.innerTile) {
      totOps *= it;
    }

    double kCacheLatency = 125.0;
    double kMemoryLatency = 420.0;
    double totMemIO = (totAccesses - totCacheMiss) * kCacheLatency +
                      totCacheMiss * kMemoryLatency;

    double myCost = totMemIO / totOps;

    // TODO: fixed sizeof
    int64_t totMemory = totRegisters * 4;
    double computePerMemory =
        static_cast<double>(totOps) / static_cast<double>(totMemory);

    int64_t kCacheSize = 2359296;
    double kMemoryBoundedThreshold = 14.0;
    bool is_mem_bounded = (totMemory * groups) > kCacheSize ||
                          computePerMemory <= kMemoryBoundedThreshold;
    // bool replace = !is_mem_bounded && best_is_mem_bounded;
    // if (is_mem_bounded == best_is_mem_bounded) {
    //   replace = is_mem_bounded
    //                 ? (myCost < bestCost)
    //                 : (groups / totMemIO > bestGroups / bestTotMemIO);
    // }

    valid_plans++;
    // semi scan
    if (diag.next() == util::DiagnosticCounter::Result::Match) {
      tileinfo += std::to_string(totRegisters);
      tileinfo += ",";
      tileinfo += std::to_string(totAccesses);
      tileinfo += ",";
      tileinfo += std::to_string(totCacheMiss);
      tileinfo += ",";
      tileinfo += std::to_string(groups);
      tileinfo += ",";
      tileinfo += std::to_string(v1cost);
      tileinfo += ",";
      tileinfo += std::to_string(totOps);
      tileinfo += ",";
      tileinfo += std::to_string(totMemIO);
      tileinfo += ",";
      tileinfo += std::to_string(myCost);
      tileinfo += ",";
      tileinfo += std::to_string(totMemory);
      tileinfo += ",";
      tileinfo += std::to_string(computePerMemory);
      tileinfo += ",";
      tileinfo += std::to_string(totMemory * groups);
      tileinfo += ",";
      if (is_mem_bounded) {
        tileinfo += "yes";
      } else {
        tileinfo += "no";
      }
      tileinfo += ",";

      std::ofstream csv("/home/adstraw/temp/subgroups.csv", std::ios::app);
      csv << plan;
      csv << tileinfo;
      csv.close();

      // replace = true;
      return -std::numeric_limits<double>::infinity();
      // groups = std::numeric_limits<int64_t>::max();
      // totMemIO = 1.0;
    }

    /*
    if (replace) {
      bestPlan = plan;
      bestCost = myCost;
      bestGroups = groups;
      bestTotMemIO = totMemIO;
    }
    */

    return myCost;
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

  util::DiagnosticCounter diag;
  uint64_t valid_plans{0};
  uint64_t total_plans{0};

  bool best_is_mem_bounded{true};
  int64_t bestGroups{0};
  double bestTotMemIO{1.0};
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
    auto mrpt_env = llvm::sys::Process::GetEnv("MAX_REGS_PER_THREAD");
    int64_t mrpt = 40;
    if (mrpt_env) {
      mrpt = std::atoi(mrpt_env->c_str());
    }

    SubgroupParams params = {
        {8, 16}, // Subgroup sizes to consider
        mrpt,    // Maximum register per thread
    };
    SubgroupCostModel cm(params, op);
    if (cm.bestCost == std::numeric_limits<double>::infinity()) {
      // If subgrouping fails, we tile accumulations instead to handle the other
      // cases
      tileAccumulations(op, false);
      setIntegerTag(op, subgroupSizeTag(), 1);
      return;
    }
    IVLOG(1, "best plan = " << cm.bestPlan);
    IVLOG(1, "valid plans = " << cm.valid_plans);
    IVLOG(1, "total plans = " << cm.total_plans);
    SubgroupApply(op, cm.bestPlan);
  }
};

} // namespace

std::unique_ptr<Pass> createSubgroupsPass() {
  return std::make_unique<SubgroupsPass>();
}

} // namespace pmlc::dialect::pxa
