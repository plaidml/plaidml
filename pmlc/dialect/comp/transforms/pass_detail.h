#pragma once

#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::dialect::comp {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/comp/transforms/passes.h.inc"

} // namespace pmlc::dialect::comp
