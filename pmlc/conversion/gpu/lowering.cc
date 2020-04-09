// Copyright 2020 Intel Corporation

#include "pmlc/conversion/gpu/lowering.h"

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

namespace pmlc::conversion::gpu {

// Add spv.entry_point_abi to gpufunc
struct LegalizeGpuOpForGpuLoweringPass
    : public mlir::ModulePass<LegalizeGpuOpForGpuLoweringPass> {
  void runOnModule() override;
};

void LegalizeGpuOpForGpuLoweringPass::runOnModule() {
  auto moduleOp = getModule();

  // set spv.target_env to moduleOp
  auto target_env = moduleOp.getAttrOfType<spirv::TargetEnvAttr>(
      spirv::getTargetEnvAttrName());
  if (!target_env) {
    auto triple = spirv::VerCapExtAttr::get(
        spirv::Version::V_1_0, {spirv::Capability::Shader},
        ArrayRef<spirv::Extension>(
            spirv::Extension::SPV_KHR_storage_buffer_storage_class),
        &getContext());
    moduleOp.setAttr(
        spirv::getTargetEnvAttrName(),
        spirv::TargetEnvAttr::get(
            triple, spirv::getDefaultResourceLimits(&getContext())));
  }

  // add spv.entry_point_abi to GPUFuncOp
  moduleOp.walk([&](mlir::gpu::GPUFuncOp func) {
    auto entry_point_abi = func.getAttrOfType<spirv::EntryPointABIAttr>(
        spirv::getEntryPointABIAttrName());
    if (!entry_point_abi) {
      // TODO local sizes should be set according to gpu.block_size, waiting for
      // upstream update
      auto entryPointAbiAttr =
          spirv::getEntryPointABIAttr({32, 1, 1}, func.getContext());
      func.setAttr(spirv::getEntryPointABIAttrName(), entryPointAbiAttr);
      return;
    }
  });
}

std::unique_ptr<mlir::Pass> createLegalizeGpuOpForGpuLoweringPass() {
  return std::make_unique<LegalizeGpuOpForGpuLoweringPass>();
}

static mlir::PassRegistration<LegalizeGpuOpForGpuLoweringPass>
    legalize_pass("legalize-gpu", "Legalize gpu.funcOp attributes");

} // namespace pmlc::conversion::gpu
