// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "pmlc/conversion/gpu/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu {
namespace gpu = mlir::gpu;

class GatherGpuLaunchFuncsPass
    : public GatherGpuLaunchFuncsPassBase<GatherGpuLaunchFuncsPass> {
public:
  void runOnOperation() override;

private:
  void gatherAllocOpsBetweenConsecutiveLaunchOps(
      mlir::Block &block,
      std::vector<std::vector<mlir::AllocOp>> &allocOpVectors,
      std::vector<gpu::LaunchFuncOp> &firstLaunchOpVector);
};

void GatherGpuLaunchFuncsPass::gatherAllocOpsBetweenConsecutiveLaunchOps(
    mlir::Block &block, std::vector<std::vector<mlir::AllocOp>> &allocOpVectors,
    std::vector<gpu::LaunchFuncOp> &firstLaunchOpVector) {
  auto &opList = block.getOperations();
  for (size_t i = 0; i < opList.size(); i++) {
    auto currOp = std::next(opList.begin(), i);
    if (mlir::isa<gpu::LaunchFuncOp>(currOp)) {
      auto launchOp = mlir::cast<gpu::LaunchFuncOp>(*currOp);
      std::vector<mlir::AllocOp> allocOpVector;
      for (size_t j = i + 1; j < opList.size(); j++) {
        auto nextOp = std::next(opList.begin(), j);
        if (mlir::isa<mlir::AllocOp>(nextOp)) {
          allocOpVector.push_back(mlir::cast<mlir::AllocOp>(*nextOp));
        } else if (!mlir::isa<gpu::LaunchFuncOp>(nextOp)) {
          firstLaunchOpVector.push_back(launchOp);
          allocOpVectors.push_back(allocOpVector);
          i = j;
          break;
        }
      }
    }
  }
}

void GatherGpuLaunchFuncsPass::runOnOperation() {
  auto module = getOperation();
  module.walk([&](mlir::FuncOp op) {
    if (op.isExternal()) {
      return;
    }
    auto &blocks = op.getBlocks();
    for (auto &block : blocks) {
      std::vector<std::vector<mlir::AllocOp>> allocOpVectors;
      std::vector<gpu::LaunchFuncOp> firstLaunchOpVector;
      gatherAllocOpsBetweenConsecutiveLaunchOps(block, allocOpVectors,
                                                firstLaunchOpVector);
      for (size_t i = 0; i < firstLaunchOpVector.size(); i++) {
        for (size_t j = 0; j < allocOpVectors[i].size(); j++) {
          allocOpVectors[i][j].getOperation()->moveBefore(
              firstLaunchOpVector[i].getOperation());
        }
      }
    }
  });
}

std::unique_ptr<mlir::Pass> createGatherGpuLaunchFuncsPass() {
  return std::make_unique<GatherGpuLaunchFuncsPass>();
}
} // namespace pmlc::conversion::gpu
