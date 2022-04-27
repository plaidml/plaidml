#pragma once

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::dialect::linalgx {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/linalgx/transforms/passes.h.inc"

} // namespace pmlc::dialect::linalgx
