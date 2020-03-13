// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::legalize_gpu {

std::unique_ptr<mlir::Pass> createLegalizeGpuOpForGpuLoweringPass();

} // namespace pmlc::conversion::legalize_gpu
