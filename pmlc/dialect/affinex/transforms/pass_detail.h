#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::affinex {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/affinex/transforms/passes.h.inc"

} // namespace pmlc::dialect::affinex
