// Copyright 2020, Intel Corporation
#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::comp_to_llvm {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/comp_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::comp_to_llvm
