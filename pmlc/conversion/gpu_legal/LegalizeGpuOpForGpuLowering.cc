// Copyright 2020 Intel Corporation

#include "pmlc/conversion/gpu_legal/LegalizeGpuOpForGpuLowering.h"
#include <vector>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::legalize_gpu {

// Add spv.entry_point_abi to gpufunc
struct LegalizeGpuOpForGpuLoweringPass
    : public mlir::OperationPass<LegalizeGpuOpForGpuLoweringPass> {
  void runOnOperation() override;
};

void LegalizeGpuOpForGpuLoweringPass::runOnOperation() {
  auto op = getOperation();

  op->walk([&](gpu::GPUFuncOp func) {
    if (auto attr = func.getAttrOfType<spirv::EntryPointABIAttr>(
            spirv::getEntryPointABIAttrName())) {
      return;
    }

    // TODO local sizes should be set according to gpu.block_size, waiting for
    // upstream update
    std::vector<int32_t> local_sizes;
    local_sizes.push_back(3);
    local_sizes.push_back(1);
    local_sizes.push_back(1);

    auto entryPointAbiAttr = spirv::getEntryPointABIAttr(
        llvm::makeArrayRef(local_sizes), func.getContext());
    func.setAttr(spirv::getEntryPointABIAttrName(), entryPointAbiAttr);
    return;
  });
}

std::unique_ptr<mlir::Pass> createLegalizeGpuOpForGpuLoweringPass() {
  return std::make_unique<LegalizeGpuOpForGpuLoweringPass>();
}

static mlir::PassRegistration<LegalizeGpuOpForGpuLoweringPass>
    legalize_pass("legalize-gpu", "Legalize gpu.funcOp attributes");

} // namespace pmlc::conversion::legalize_gpu
