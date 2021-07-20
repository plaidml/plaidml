// Copyright 2021 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::pml {

std::unique_ptr<mlir::Pass> createApplyRulesPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/pml/transforms/passes.h.inc"

} // namespace pmlc::dialect::pml
