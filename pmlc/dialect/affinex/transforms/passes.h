// Copyright 2020 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::affinex {

std::unique_ptr<mlir::Pass>
createAffinexLoopUnroll(uint64_t operationLimit = 256);
std::unique_ptr<mlir::Pass> createAffinexMemRefDataFlowOpt();
std::unique_ptr<mlir::Pass> createAffinexDeadMemRefElimination();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/affinex/transforms/passes.h.inc"

} // namespace pmlc::dialect::affinex
