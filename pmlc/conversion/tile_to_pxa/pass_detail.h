#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::tile_to_pxa {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/tile_to_pxa/passes.h.inc"

} // namespace pmlc::conversion::tile_to_pxa
