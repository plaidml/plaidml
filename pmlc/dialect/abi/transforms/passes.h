// Copyright 2020 Intel Corporation
#pragma once

#include <memory>

#include "mlir/Pass/PassRegistry.h"

#include "pmlc/dialect/abi/ir/dialect.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace pmlc::dialect::abi {

std::unique_ptr<mlir::Pass> createLowerToABIPass();

#define GEN_PASS_REGISTRATION
#include "pmlc/dialect/abi/transforms/passes.h.inc"

} // namespace pmlc::dialect::abi
