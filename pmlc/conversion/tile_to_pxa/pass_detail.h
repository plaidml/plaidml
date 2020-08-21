#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/stdx/ir/ops.h"

namespace pmlc::conversion::tile_to_pxa {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/tile_to_pxa/passes.h.inc"

} // namespace pmlc::conversion::tile_to_pxa
