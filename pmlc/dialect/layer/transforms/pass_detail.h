// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::dialect::layer {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/layer/transforms/passes.h.inc"

} // namespace pmlc::dialect::layer
