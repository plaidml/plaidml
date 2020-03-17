// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace pmlc::conversion::kernel_outlining {

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
createGpuKernelOutliningPass();

} // namespace pmlc::conversion::kernel_outlining
