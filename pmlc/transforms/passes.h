// Copyright 2020 Intel Corporation
#pragma once

#include <memory>

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::transforms {

std::unique_ptr<mlir::Pass> createHoistingPass();

#define GEN_PASS_REGISTRATION
#include "pmlc/transforms/passes.h.inc"

} // namespace pmlc::transforms
