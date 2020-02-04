// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::stdx_to_llvm {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace pmlc::conversion::stdx_to_llvm
