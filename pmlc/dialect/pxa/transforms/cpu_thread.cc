// Copyright 2020 Intel Corporation

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/tile.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/tags.h"

using namespace mlir; // NOLINT

#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "llvm/Support/JSON.h"
namespace pmlc::dialect::pxa {

namespace {

static constexpr llvm::StringLiteral kCpuThreadTag = "cpuThread";

// Pick the tiling that is as large as possible without going over maxThreads
struct CostModel {
  unsigned maxThreads;
  ArrayRef<int64_t> strides;

  explicit CostModel(unsigned maxThreads, ArrayRef<int64_t> strides)
      : maxThreads(maxThreads), strides(strides) {}

  double operator()(ArrayRef<int64_t> tile, double bestCost) const {
    int64_t innerSize = 1;
    int64_t maxStride = 1;
    for (size_t i = 0; i < tile.size(); i++) {
      innerSize *= tile[i];
      if (tile[i] != 1)
        maxStride = std::max(maxStride, strides[i]);
    }
    if (innerSize > maxThreads) {
      return std::numeric_limits<double>::infinity();
    }
    return (1.0 / innerSize) + (1.0 / maxStride);
  }
};

typedef struct ThreadSchedParams {
  std::string schedVal;
  int chunkSize;
  std::string schedModifier;
  int collapseVal;
} ThreadSchedParams;

struct CPUThreadPass : public CPUThreadBase<CPUThreadPass> {
  CPUThreadPass() = default;
  explicit CPUThreadPass(unsigned threads) { this->threads = threads; }

  void runOnFunction() final {
    auto func = getFunction();
    std::list<AffineParallelOp> opStack;
    std::vector<int> opId;
    std::map<Operation *, std::string> opIdMap;
    std::map<std::string, ThreadSchedParams *> schedParamsMap;

    func.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      // Assign a unique id to each affineparallel op
      int lastUsedIdAtLevel = 0;
      while (!opStack.empty() &&
             opStack.back().getOperation() !=
                 op.getBody()->getParentOp()->getParentOp()) {
        opStack.pop_back();
        if (opStack.empty()) {
          // TODO Hack
          // At the topmost level, maintain the same starting id (0)
          lastUsedIdAtLevel = opId.back();
        } else {
          lastUsedIdAtLevel = opId.back() + 1;
        }
        opId.pop_back();
      }
      opStack.push_back(op);
      opId.push_back(lastUsedIdAtLevel);
      if (opStack.front().getNumDims() != 7) {
        return WalkResult::skip();
      }
      std::stringstream idString("");
      for (int i = 0; i < opId.size(); i++) {
        idString << opId[i];
        if (i < opId.size() - 1) {
          idString << ",";
        }
      }
      opIdMap.insert(std::make_pair(op.getOperation(), idString.str()));
    });

    // Read the file specified by the  environment variable for loop
    // configuration
    if (!util::getEnvVar("PLAIDML_THREAD_DIST_CONFIG_FILE").empty()) {
      auto configFileName = util::getEnvVar("PLAIDML_THREAD_DIST_CONFIG_FILE");
      IVLOG(1, "Configuration file name:" << configFileName);
      std::ifstream configFile(configFileName, std::ifstream::binary);
      std::stringstream jsonStringStream;
      jsonStringStream << configFile.rdbuf();
      std::string jsonString = jsonStringStream.str();
      llvm::Expected<llvm::json::Value> threadMapping =
          llvm::json::parse(llvm::StringRef(jsonString));
      if (threadMapping.takeError()) {
        IVLOG(1, "Error");
      } else if (llvm::json::Array *loopSchedObj =
                     threadMapping->getAsObject()->getArray("loopschedule")) {
        llvm::json::Array loopSched = *loopSchedObj;
        for (int i = 0; i < loopSched.size(); i++) {
          ThreadSchedParams *params = new ThreadSchedParams();
          auto id =
              loopSched[i].getAsObject()->getString("id").getValue().str();
          params->schedVal = loopSched[i]
                                 .getAsObject()
                                 ->getString("sched_val")
                                 .getValue()
                                 .str();
          params->chunkSize = std::stoi(loopSched[i]
                                            .getAsObject()
                                            ->getString("sched_chunk")
                                            .getValue()
                                            .str());
          params->collapseVal = std::stoi(loopSched[i]
                                              .getAsObject()
                                              ->getString("collapse_val")
                                              .getValue()
                                              .str());
          schedParamsMap[id] = params;
        }
      }
    }
  }

  void processOp(AffineParallelOp op, ThreadSchedParams *schedParams) {
    if (schedParams) {
      setIntegerTag(op, "collapse", schedParams->collapseVal);
      setIntegerTag(op, "chunk_size", schedParams->chunkSize);
      setUnitTag(op, schedParams->schedVal);
      setUnitTag(op, kCpuThreadTag);
      return;
    }

    auto maybeRanges = op.getConstantRanges();
    if (!maybeRanges) {
      // Fail if we can't compute the ranges at compile time
      return;
    }

    SmallVector<int64_t> strides(op.getNumDims(), 0);
    if (auto lastWriter =
            dyn_cast_or_null<PxaReduceOp>(getPrevWriter(op.getResult(0)))) {
      if (Optional<StrideInfo> si = computeStrideInfo(lastWriter)) {
        for (BlockArgument arg : op.getIVs()) {
          strides[arg.getArgNumber()] = si->strides[arg];
        }
      }
    }

    CostModel model(threads, strides);
    auto tileSize =
        findBestTileSize(EvenTilingGenerator(), model, *maybeRanges);
    // Invert tiling (we want 'threads' on the outer loop
    for (size_t i = 0; i < tileSize.size(); i++) {
      tileSize[i] = (*maybeRanges)[i] / tileSize[i];
    }
    // Tile and tag
    performTiling(op, tileSize);
    setUnitTag(op, kCpuThreadTag);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createCPUThreadPass() {
  return std::make_unique<CPUThreadPass>();
}

std::unique_ptr<mlir::Pass> createCPUThreadPass(unsigned threads) {
  return std::make_unique<CPUThreadPass>(threads);
}

} // namespace pmlc::dialect::pxa.
