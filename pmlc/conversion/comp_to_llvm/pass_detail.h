// Copyright 2020, Intel Corporation
#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::comp_to_llvm {

/// Information about serialized module.
struct BinaryModuleInfo {
  /// Global operation containing binary content of serialized module.
  mlir::LLVM::GlobalOp symbol;
  /// Size of serialized module in bytes.
  size_t bytes;
  /// Map from kernel name to global symbol containing that kernel's name.
  /// Names are stored as null terminated char arrays.
  std::map<std::string, mlir::LLVM::GlobalOp> kernelsNameMap;
};

/// Map from module name to information about serialized binary.
class BinaryModulesMap : public std::map<std::string, BinaryModuleInfo> {};

#define GEN_PASS_CLASSES
#include "pmlc/conversion/comp_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::comp_to_llvm
