// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::gpu {

// Set spv.target_env and spv.entry_point_abi
std::unique_ptr<mlir::Pass> createGpuKernelOutliningPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/gpu/passes.h.inc"

} // namespace pmlc::conversion::gpu
