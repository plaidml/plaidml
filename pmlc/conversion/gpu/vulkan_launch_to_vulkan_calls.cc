//===- ConvertLaunchFuncToVulkanCalls.cpp - MLIR Vulkan conversion passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert vulkan launch call into a sequence of
// Vulkan runtime calls. The Vulkan runtime API surface is huge so currently we
// don't expose separate external functions in IR for each of them, instead we
// expose a few external functions to wrapper libraries which manages Vulkan
// runtime.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallString.h"

using namespace mlir; // NOLINT[build/namespaces]

static constexpr const char *kSPIRVBinary = "SPIRV_BIN";
static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";
static constexpr const char *kPrint_memref_f32 = "print_memref_f32";
static constexpr const char *kBindBufferFloat32 = "bindBufferFloat32";
static constexpr const char *kBindBufferInt64 = "bindBufferInt64";
static constexpr const char *kConfigureLaunchKernelAction =
    "configureLaunchKernelAction";

namespace {
class VulkanLaunchFuncToVulkanCallsPass
    : public ModulePass<VulkanLaunchFuncToVulkanCallsPass> {
private:
  LLVM::LLVMDialect *getLLVMDialect() { return llvmDialect; }

  llvm::LLVMContext &getLLVMContext() {
    return getLLVMDialect()->getLLVMContext();
  }

  void initializeCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmFloatType = LLVM::LLVMType::getFloatTy(llvmDialect);
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);
  }

  LLVM::LLVMType getFloatType() { return llvmFloatType; }
  LLVM::LLVMType getVoidType() { return llvmVoidType; }
  LLVM::LLVMType getPointerType() { return llvmPointerType; }
  LLVM::LLVMType getInt32Type() { return llvmInt32Type; }
  LLVM::LLVMType getInt64Type() { return llvmInt64Type; }

  /// Creates a LLVM global for the given `name`.
  Value createEntryPointNameConstant(StringRef name, Location loc,
                                     OpBuilder &builder);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Checks whether the given LLVM::CallOp is a vulkan launch call op.
  bool isVulkanLaunchCallOp(LLVM::CallOp callOp) {
    return (callOp.callee() &&
            callOp.callee().getValue().startswith(kVulkanLaunch) &&
            callOp.getNumOperands() >= gpu::LaunchOp::kNumConfigOperands);
  }
  /// Translates the given `vulkanLaunchCallOp` to the sequence of Vulkan
  /// runtime calls.
  void translateVulkanLaunchCall(LLVM::CallOp vulkanLaunchCallOp);

  /// Creates call to `bindMemRef` for each memref operand.
  void createBindMemRefCalls(LLVM::CallOp vulkanLaunchCallOp,
                             Value vulkanRuntime);

  /// Collects SPIRV attributes from the given `vulkanLaunchCallOp`.
  void collectSPIRVAttributes(LLVM::CallOp vulkanLaunchCallOp);

public:
  void runOnModule() override;

private:
  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmFloatType;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;

  size_t spv_entry_index = 0;
  size_t spv_binary_index = 0;

  SmallVector<std::pair<StringAttr, StringAttr>, 1> spirvAttributes;
};

} // anonymous namespace

void VulkanLaunchFuncToVulkanCallsPass::runOnModule() {
  initializeCachedTypes();

  // Collect SPIR-V attributes such as `spirv_blob` and
  // `spirv_entry_point_name`.
  getModule().walk([this](LLVM::CallOp op) {
    if (isVulkanLaunchCallOp(op))
      collectSPIRVAttributes(op);
  });

  // Convert vulkan launch call op into a sequence of Vulkan runtime calls.
  getModule().walk([this](LLVM::CallOp op) {
    if (isVulkanLaunchCallOp(op))
      translateVulkanLaunchCall(op);
  });

  // clear VulanLaunch and change print_memref_f32 declaration
  for (auto func :
       llvm::make_early_inc_range(getModule().getOps<LLVM::LLVMFuncOp>())) {
    if (func.getName().startswith(kVulkanLaunch)) {
      func.erase();
    } else if (func.getName().equals(kPrint_memref_f32)) {
      OpBuilder builder(getModule().getBody()->getTerminator());
      Location loc = func.getLoc();
      func.erase();
      builder.create<LLVM::LLVMFuncOp>(
          loc, kPrint_memref_f32,
          LLVM::LLVMType::getFunctionTy(getVoidType(),
                                        {getInt64Type(), getPointerType()},
                                        /*isVarArg=*/false));
    }
  }

  // change kBindBufferFloat32 declaration
  for (auto func :
       llvm::make_early_inc_range(getModule().getOps<LLVM::LLVMFuncOp>())) {
    if (func.getName().equals(kBindBufferFloat32)) {
      OpBuilder builder(getModule().getBody()->getTerminator());
      Location loc = func.getLoc();
      func.erase();
      builder.create<LLVM::LLVMFuncOp>(
          loc, kBindBufferFloat32,
          LLVM::LLVMType::getFunctionTy(getVoidType(),
                                        {getPointerType(), getInt32Type(),
                                         getInt32Type(), getInt64Type(),
                                         getPointerType()},
                                        /*isVarArg=*/false));
    }
  }

  // change kBindBufferInt64 declaration
  for (auto func :
       llvm::make_early_inc_range(getModule().getOps<LLVM::LLVMFuncOp>())) {
    if (func.getName().equals(kBindBufferInt64)) {
      OpBuilder builder(getModule().getBody()->getTerminator());
      Location loc = func.getLoc();
      func.erase();
      builder.create<LLVM::LLVMFuncOp>(
          loc, kBindBufferInt64,
          LLVM::LLVMType::getFunctionTy(getVoidType(),
                                        {getPointerType(), getInt32Type(),
                                         getInt32Type(), getInt64Type(),
                                         getPointerType()},
                                        /*isVarArg=*/false));
    }
  }
}

void VulkanLaunchFuncToVulkanCallsPass::collectSPIRVAttributes(
    LLVM::CallOp vulkanLaunchCallOp) {
  // Check that `kSPIRVBinary` and `kSPIRVEntryPoint` are present in
  // attributes for the given vulkan launch call.
  auto spirvBlobAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVBlobAttrName);
  if (!spirvBlobAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVBlobAttrName << " attribute";
    return signalPassFailure();
  }

  auto spirvEntryPointNameAttr =
      vulkanLaunchCallOp.getAttrOfType<StringAttr>(kSPIRVEntryPointAttrName);
  if (!spirvEntryPointNameAttr) {
    vulkanLaunchCallOp.emitError()
        << "missing " << kSPIRVEntryPointAttrName << " attribute";
    return signalPassFailure();
  }

  spirvAttributes.push_back(
      std::make_pair(spirvBlobAttr, spirvEntryPointNameAttr));
}

void VulkanLaunchFuncToVulkanCallsPass::declareVulkanFunctions(Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kConfigureLaunchKernelAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kConfigureLaunchKernelAction,
        LLVM::LLVMType::getFunctionTy(
            getVoidType(),
            {getPointerType(), getPointerType(), getInt32Type(),
             getPointerType(), getInt64Type(), getInt64Type(), getInt64Type()},
            /*isVarArg=*/false));
  }
}

Value VulkanLaunchFuncToVulkanCallsPass::createEntryPointNameConstant(
    StringRef name, Location loc, OpBuilder &builder) {
  SmallString<16> shaderName(name.begin(), name.end());
  // Append `\0` to follow C style string given that
  // LLVM::createGlobalString() won't handle this directly for us.
  shaderName.push_back('\0');

  std::string entryPointGlobalName =
      (name + "_spv_entry_point_name" + std::to_string(spv_entry_index)).str();
  spv_entry_index++;
  return LLVM::createGlobalString(loc, builder, entryPointGlobalName,
                                  shaderName, LLVM::Linkage::Internal,
                                  getLLVMDialect());
}

void VulkanLaunchFuncToVulkanCallsPass::translateVulkanLaunchCall(
    LLVM::CallOp vulkanLaunchCallOp) {
  OpBuilder builder(vulkanLaunchCallOp);
  Location loc = vulkanLaunchCallOp.getLoc();

  // The first operand of vulkanLaunchCallOp is a pointer to Vulkan runtime, we
  // need to pass that pointer to each Vulkan runtime call.
  auto vulkanRuntime = vulkanLaunchCallOp.getOperand(0);

  // Create LLVM global with SPIR-V binary data, so we can pass a pointer with
  // that data to runtime call.
  Value ptrToSPIRVBinary = LLVM::createGlobalString(
      loc, builder, kSPIRVBinary + std::to_string(spv_binary_index),
      spirvAttributes[spv_binary_index].first.getValue(),
      LLVM::Linkage::Internal, getLLVMDialect());

  // Create LLVM constant for the size of SPIR-V binary shader.
  Value binarySize = builder.create<LLVM::ConstantOp>(
      loc, getInt32Type(),
      builder.getI32IntegerAttr(
          spirvAttributes[spv_binary_index].first.getValue().size()));

  // Create LLVM global with entry point name.
  Value entryPointName = createEntryPointNameConstant(
      spirvAttributes[spv_binary_index].second.getValue(), loc, builder);

  // Create configureLaunchKernelAction.
  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{getVoidType()},
      builder.getSymbolRefAttr(kConfigureLaunchKernelAction),
      ArrayRef<Value>{vulkanRuntime, ptrToSPIRVBinary, binarySize,
                      entryPointName, vulkanLaunchCallOp.getOperand(1),
                      vulkanLaunchCallOp.getOperand(2),
                      vulkanLaunchCallOp.getOperand(3)});

  declareVulkanFunctions(loc);
  vulkanLaunchCallOp.erase();
  spv_binary_index++;
}

namespace pmlc::conversion::gpu {
std::unique_ptr<mlir::Pass> createConvertVulkanLaunchFuncToVulkanCallsPass() {
  return std::make_unique<VulkanLaunchFuncToVulkanCallsPass>();
}
} // namespace pmlc::conversion::gpu
static PassRegistration<VulkanLaunchFuncToVulkanCallsPass>
    pass("pmlc-launch-func-to-vulkan",
         "Convert vulkanLaunch external call to Vulkan runtime external calls");
