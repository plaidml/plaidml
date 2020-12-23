// Copyright 2020, Intel Corporation
#pragma once

#include <memory>
#include <string>

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class OwningRewritePatternList;
class Pass;
class TypeConverter;
} // namespace mlir

namespace pmlc::conversion::comp_to_llvm {
class BinaryModulesMap;

/// Creates empty BinaryModulesMap that can be filled by functions
/// serializing modules to binary form.
std::unique_ptr<BinaryModulesMap> getEmptyModulesMap();

/// Walks operation `op` and serializes all spirv modules,
/// inserting binary as constants into `op`s region.
/// Fills `map` with information about serialized modules.
/// Erases all serialized spirv modules.
mlir::LogicalResult serializeSpirvKernels(mlir::ModuleOp &op,
                                          BinaryModulesMap &map);

std::unique_ptr<mlir::Pass> createConvertCompToLLVMPass();
std::unique_ptr<mlir::Pass>
createConvertCompToLLVMPass(const std::string &prefix);

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/comp_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::comp_to_llvm
