// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu_to_vulkan {

std::unique_ptr<mlir::Pass> createLegalizeGpuOpForGpuLoweringPass();

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanCallsPass();

std::unique_ptr<mlir::Pass> createConvertVulkanLaunchFuncToVulkanCallsPass();

std::unique_ptr<mlir::Pass> createLowerGpuToVulkanCallsPass();

} // namespace pmlc::conversion::gpu_to_vulkan
