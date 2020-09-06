// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::stdx {

std::unique_ptr<mlir::Pass> createBoundsCheckPass();

std::unique_ptr<mlir::Pass> createI1StorageToI32Pass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/stdx/transforms/passes.h.inc"

} // namespace pmlc::dialect::stdx
