// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

#include "pmlc/dialect/comp/ir/dialect.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu {

// Find all GPU loops + convert to kernels
std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass();
std::unique_ptr<mlir::Pass>
createGpuKernelOutliningPass(pmlc::dialect::comp::ExecEnvRuntime runtime,
                             unsigned memorySpace);

// Gather gpu.launch_func Ops separated by std.alloc
std::unique_ptr<mlir::Pass> createGatherGpuLaunchFuncsPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/gpu/passes.h.inc"

} // namespace pmlc::conversion::gpu
