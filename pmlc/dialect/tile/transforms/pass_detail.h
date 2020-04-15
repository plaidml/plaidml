#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::tile {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/tile/transforms/passes.h.inc"

} // namespace pmlc::dialect::tile
