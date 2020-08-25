#pragma once

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::target::intel_gen {

#define GEN_PASS_CLASSES
#include "pmlc/target/intel_gen/passes.h.inc"

} // namespace pmlc::target::intel_gen
