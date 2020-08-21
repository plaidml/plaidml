#pragma once

#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::pxa {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/pxa/transforms/passes.h.inc"

} // namespace pmlc::dialect::pxa
