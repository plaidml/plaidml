// Copyright 2020, Intel Corporation
#pragma once

#include <memory>

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

// ============================================================================
// Common
// ============================================================================
/// Creates empty BinaryModulesMap that can be filled by functions
/// serializing modules to binary form.
std::unique_ptr<BinaryModulesMap> getEmptyModulesMap();

/// Walks operation `op` and serializes all spirv modules,
/// inserting binary as constants into `op`s region.
/// Fills `map` with information about serialized modules.
/// Erases all serialized spirv modules.
mlir::LogicalResult serializeSpirvKernels(mlir::ModuleOp &op,
                                          BinaryModulesMap &map);

/// Populates type and operation conversion patterns that are common to all
/// comp lowerings.
void populateCommonPatterns(mlir::MLIRContext *context,
                            mlir::TypeConverter &typeConverter,
                            mlir::TypeConverter &signatureConverter,
                            mlir::OwningRewritePatternList &patterns);

/// Adds declarations of functions common across lowerings
/// to specified top-level module.
void addCommonFunctionDeclarations(mlir::ModuleOp &module);

// ============================================================================
// OpenCL
// ============================================================================
/// Populates `patterns` and `typeConverter` with conversion patterns that
/// perform lowering for OpenCL runtime.
void populateCompToOclPatterns(mlir::MLIRContext *context,
                               const BinaryModulesMap &modulesMap,
                               mlir::TypeConverter &typeConverter,
                               mlir::OwningRewritePatternList &patterns);

/// Adds declarations of functions specific to OpenCL runtime.
void addOclFunctionDeclarations(mlir::ModuleOp &module);

/// Returns pass that will perform lowering for OpenCL runtime.
/// To provide stronger guarantees any comp operation with runtime different
/// than OpenCL will cause this pass to report failure.
std::unique_ptr<mlir::Pass> createConvertCompToOclPass();

// ============================================================================
// Vulkan
// ============================================================================

void populateCompToVkPatterns(mlir::MLIRContext *context,
                              const BinaryModulesMap &modulesMap,
                              mlir::ModuleOp moduleOp, uint32_t numKernel,
                              mlir::TypeConverter &typeConverter,
                              mlir::OwningRewritePatternList &patterns);

void addVkFunctionDeclarations(mlir::ModuleOp &module);

/// Returns pass that will perform lowering for Vulkan runtime.
/// To provide stronger guarantees any comp operation with runtime different
/// than OpenCL will cause this pass to report failure.
std::unique_ptr<mlir::Pass> createConvertCompToVulkanPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/comp_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::comp_to_llvm
