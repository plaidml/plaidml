// Copyright 2021, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::conversion::linalg_to_pxa {

std::unique_ptr<mlir::Pass> createLowerLinalgToPXAPass();

// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/linalg_to_pxa/passes.h.inc"

} // namespace pmlc::conversion::linalg_to_pxa
