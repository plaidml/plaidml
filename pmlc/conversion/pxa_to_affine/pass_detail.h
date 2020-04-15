#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::pxa_to_affine {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/pxa_to_affine/passes.h.inc"

} // namespace pmlc::conversion::pxa_to_affine
