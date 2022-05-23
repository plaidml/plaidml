// Copyright 2022 Intel Corporation

#pragma once

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

namespace pmlc::dialect::linalgx {

std::unique_ptr<mlir::Pass> createRegulateDepthwisePass();
std::unique_ptr<mlir::Pass> createNameConvVariablesPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/linalgx/transforms/passes.h.inc"

} // namespace pmlc::dialect::linalgx
