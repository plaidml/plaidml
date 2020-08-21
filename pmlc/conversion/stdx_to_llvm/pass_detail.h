#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::stdx_to_llvm {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/stdx_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::stdx_to_llvm
