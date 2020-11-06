// Copyright 2020 Intel Corporation
#pragma once

#include <memory>

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::comp {

std::unique_ptr<mlir::Pass> createExecEnvCoalescingPass();

std::unique_ptr<mlir::Pass> createMinimizeAllocationsPass();

std::unique_ptr<mlir::Pass> createRecalculateEventDepsPass();
std::unique_ptr<mlir::Pass> createRecalculateEventDepsPass(bool safeDealloc);

std::unique_ptr<mlir::Pass> createRemoveRedundantRWPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/comp/transforms/passes.h.inc"

} // namespace pmlc::dialect::comp
