// Copyright 2020, Intel Corporation
#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::target::intel_level_zero {

std::unique_ptr<mlir::Pass> createAddSpirvTargetPass();
std::unique_ptr<mlir::Pass> createAddSpirvTargetPass(unsigned sprivVersion);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/target/intel_level_zero/passes.h.inc"

} // namespace pmlc::target::intel_level_zero
