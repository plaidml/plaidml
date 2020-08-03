// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::comp {

std::unique_ptr<mlir::Pass> createConvertGpuToCompPass();

std::unique_ptr<mlir::Pass> createConvertCompToOpenClPass();

} // namespace pmlc::conversion::comp
