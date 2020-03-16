// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu {

std::unique_ptr<mlir::Pass> createLegalizeGpuOpForGpuLoweringPass();

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanCallsPass();

} // namespace pmlc::conversion::gpu
