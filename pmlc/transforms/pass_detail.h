// Copyright 2021 Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::transforms {

#define GEN_PASS_CLASSES
#include "pmlc/transforms/passes.h.inc"

} // namespace pmlc::transforms
