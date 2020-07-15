// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu {

// Set spv.target_env and spv.entry_point_abi
std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass();

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanCallsPass();

} // namespace pmlc::conversion::gpu
