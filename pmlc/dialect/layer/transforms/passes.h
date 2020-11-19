// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::layer {

std::unique_ptr<mlir::Pass> createInlineLayersPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/layer/transforms/passes.h.inc"

} // namespace pmlc::dialect::layer
