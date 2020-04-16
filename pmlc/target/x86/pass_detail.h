#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::target::x86 {

#define GEN_PASS_CLASSES
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
