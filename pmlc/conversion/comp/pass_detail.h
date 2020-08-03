#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::comp {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/comp/passes.h.inc"

} // namespace pmlc::conversion::comp
