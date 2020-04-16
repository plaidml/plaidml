#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::stdx {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/stdx/transforms/passes.h.inc"

} // namespace pmlc::dialect::stdx
