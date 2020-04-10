//===- ConvertGPULaunchFuncToVulkanLaunchFunc.cpp - MLIR conversion pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert gpu launch function into a vulkan
// launch function. Creates a SPIR-V binary shader from the `spirv::ModuleOp`
// using `spirv::serialize` function, attaches binary data and entry point name
// as an attributes to vulkan launch call op.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";
static constexpr const char *kPrint_memref_f32 = "print_memref_f32";
static constexpr const char *kInitVulkan = "initVulkan";
static constexpr const char *kSubmitCommandBuffers = "submitCommandBuffers";
static constexpr const char *kDeinitVulkan = "deinitVulkan";
static constexpr const char *kCreateLaunchKernelAction =
    "createLaunchKernelAction";
static constexpr const char *kCreateMemoryTransferAction =
    "createMemoryTransferAction";
static constexpr const char *kAddVulkanLaunchActionToSchedule =
    "addVulkanLaunchActionToSchedule";
static constexpr const char *kSetLaunchKernelAction = "setLaunchKernelAction";
static constexpr const char *kBindBufferFloat32 = "bindBufferFloat32";
static constexpr const char *kBindBufferInt64 = "bindBufferInt64";

namespace {
/// A pass to convert gpu launch op to vulkan launch call op, by creating a
/// SPIR-V binary shader from `spirv::ModuleOp` using `spirv::serialize`
/// function and attaching binary data and entry point name as an attributes to
/// created vulkan launch call op.
class ConvertGpuLaunchFuncToVulkanLaunchFunc
    : public ModulePass<ConvertGpuLaunchFuncToVulkanLaunchFunc> {
public:
  void runOnModule() override;

private:
  /// Creates a SPIR-V binary shader from the given `module` using
  /// `spirv::serialize` function.
  LogicalResult createBinaryShader(ModuleOp module,
                                   std::vector<char> &binaryShader);

  LogicalResult printLauchOpBuffers(Location loc, OpBuilder &builder,
                                    mlir::gpu::LaunchFuncOp launchOp);

  /// Declares all needed runtime functions.
  void declareVulkanFunctions(Location loc);

  /// Converts the given `luanchOp` to vulkan launch call.
  void convertGpuLaunchFunc(gpu::LaunchFuncOp launchOp);

  /// Checks where the given type is supported by Vulkan runtime.
  bool isSupportedType(Type type) {
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>())
      return memRefType.hasRank() &&
             (memRefType.getRank() >= 1 && memRefType.getRank() <= 3);
    return false;
  }

  void getCachedTypes() {
    llvmDialect = getContext().getRegisteredDialect<LLVM::LLVMDialect>();
    llvmFloatType = LLVM::LLVMType::getFloatTy(llvmDialect);
    llvmVoidType = LLVM::LLVMType::getVoidTy(llvmDialect);
    llvmPointerType = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    llvmInt32Type = LLVM::LLVMType::getInt32Ty(llvmDialect);
    llvmInt64Type = LLVM::LLVMType::getInt64Ty(llvmDialect);

    OpBuilder builder(getModule());
    mlirInt32Type = builder.getIntegerType(32);
    mlirInt64Type = builder.getIntegerType(64);
    mlirFloat32Type = builder.getF32Type();
    mlirUnrankedMemRefF32Type =
        UnrankedMemRefType::get(mlirFloat32Type, /*memorySpace=*/0);
    mlirUnrankedMemRefI64Type =
        UnrankedMemRefType::get(mlirInt64Type, /*memorySpace=*/0);

    mlir1DDynamicMemRefF32Type = getMemRefFloat32Type({-1});
    mlir2DDynamicMemRefF32Type = getMemRefFloat32Type({-1, -1});
    mlir3DDynamicMemRefF32Type = getMemRefFloat32Type({-1, -1, -1});

    mlir1DDynamicMemRefI64Type = getMemRefInt64Type({-1});
    mlir2DDynamicMemRefI64Type = getMemRefInt64Type({-1, -1});
    mlir3DDynamicMemRefI64Type = getMemRefInt64Type({-1, -1, -1});
  }

  mlir::Type getMemRefFloat32Type(SmallVector<int64_t, 4> shape) {
    return MemRefType::get(shape, mlirFloat32Type);
  }

  mlir::Type getMemRefInt64Type(SmallVector<int64_t, 4> shape) {
    return MemRefType::get(shape, mlirInt64Type);
  }
  /// Declares the vulkan launch function. Returns an error if the any type of
  /// operand is unsupported by Vulkan runtime.
  LogicalResult declareVulkanLaunchFunc(Location loc,
                                        gpu::LaunchFuncOp launchOp);

  LLVM::LLVMDialect *llvmDialect;
  LLVM::LLVMType llvmFloatType;
  LLVM::LLVMType llvmVoidType;
  LLVM::LLVMType llvmPointerType;
  LLVM::LLVMType llvmInt32Type;
  LLVM::LLVMType llvmInt64Type;

  mlir::Type mlirInt32Type;
  mlir::Type mlirInt64Type;
  mlir::Type mlirFloat32Type;

  mlir::Type mlir1DDynamicMemRefF32Type;
  mlir::Type mlir2DDynamicMemRefF32Type;
  mlir::Type mlir3DDynamicMemRefF32Type;
  mlir::Type mlirUnrankedMemRefF32Type;

  mlir::Type mlir1DDynamicMemRefI64Type;
  mlir::Type mlir2DDynamicMemRefI64Type;
  mlir::Type mlir3DDynamicMemRefI64Type;
  mlir::Type mlirUnrankedMemRefI64Type;

  mlir::Value vulkanRuntime;

  size_t numKernel = 0;
  size_t lauchFuncIndex = 0;
  llvm::DenseMap<Value, llvm::SmallVector<uint, 2>> bufferMap;
};

} // anonymous namespace

void ConvertGpuLaunchFuncToVulkanLaunchFunc::runOnModule() {
  getCachedTypes();
  getModule().walk([this](gpu::LaunchFuncOp op) { numKernel++; });
  getModule().walk([this](gpu::LaunchFuncOp op) { convertGpuLaunchFunc(op); });

  // Erase `gpu::GPUModuleOp` and `spirv::Module` operations.
  for (auto gpuModule :
       llvm::make_early_inc_range(getModule().getOps<gpu::GPUModuleOp>()))
    gpuModule.erase();

  for (auto spirvModule :
       llvm::make_early_inc_range(getModule().getOps<spirv::ModuleOp>()))
    spirvModule.erase();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::declareVulkanLaunchFunc(
    Location loc, gpu::LaunchFuncOp launchOp) {
  OpBuilder builder(getModule().getBody()->getTerminator());
  // TODO: Workgroup size is written into the kernel. So to properly modelling
  // vulkan launch, we cannot have the local workgroup size configuration here.
  SmallVector<Type, 8> vulkanLaunchTypes{launchOp.getOperandTypes()};
  vulkanLaunchTypes.insert(vulkanLaunchTypes.begin(), vulkanRuntime.getType());

  // Check that all operands have supported types except those for the launch
  // configuration.
  for (auto type : llvm::drop_begin(vulkanLaunchTypes,
                                    gpu::LaunchOp::kNumConfigOperands + 1)) {
    if (!isSupportedType(type))
      return launchOp.emitError() << type << " is unsupported to run on Vulkan";
  }

  // Declare vulkan launch function.
  builder.create<FuncOp>(
      loc, kVulkanLaunch + std::to_string(lauchFuncIndex),
      FunctionType::get(vulkanLaunchTypes, ArrayRef<Type>{}, loc->getContext()),
      ArrayRef<NamedAttribute>{});

  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::createBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {

  SmallVector<uint32_t, 0> binary;
  size_t shader_index = 0;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (shader_index == lauchFuncIndex) {
      if (failed(spirv::serialize(spirvModule, binary))) {
        return failure();
      }
    }
    shader_index++;
  }

  binaryShader.resize(binary.size() * sizeof(uint32_t));
  std::memcpy(binaryShader.data(), reinterpret_cast<char *>(binary.data()),
              binaryShader.size());
  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::printLauchOpBuffers(
    Location loc, OpBuilder &builder, mlir::gpu::LaunchFuncOp launchOp) {
  auto buffers = launchOp.operands();
  for (auto buffer : buffers) {
    auto type = buffer.getType();
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>()) {
      auto rank = memRefType.getRank();
      auto elemntType = memRefType.getElementType();
      if (elemntType.isF32()) {
        mlir::Type memRefType;
        if (rank == 1)
          memRefType = mlir1DDynamicMemRefF32Type;
        if (rank == 2)
          memRefType = mlir2DDynamicMemRefF32Type;
        if (rank == 3)
          memRefType = mlir3DDynamicMemRefF32Type;
        auto dynamicBuffer =
            builder.create<mlir::MemRefCastOp>(loc, buffer, memRefType);
        auto unrankedBuffer = builder.create<mlir::MemRefCastOp>(
            loc, dynamicBuffer, mlirUnrankedMemRefF32Type);

        builder.create<CallOp>(loc, ArrayRef<Type>{},
                               builder.getSymbolRefAttr(kPrint_memref_f32),
                               ArrayRef<Value>(unrankedBuffer));
      }
    }
  }
  return success();
}

void ConvertGpuLaunchFuncToVulkanLaunchFunc::declareVulkanFunctions(
    Location loc) {
  ModuleOp module = getModule();
  OpBuilder builder(module.getBody()->getTerminator());

  if (!module.lookupSymbol(kPrint_memref_f32)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kPrint_memref_f32,
        FunctionType::get({ArrayRef<Type>{mlirUnrankedMemRefF32Type}}, {},
                          &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kBindBufferFloat32)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kBindBufferFloat32,
        FunctionType::get(
            {ArrayRef<Type>{llvmPointerType, llvmInt32Type, llvmInt32Type,
                            mlirUnrankedMemRefF32Type}},
            {}, &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kBindBufferInt64)) {
    auto &ctx = getContext();
    builder.create<FuncOp>(
        loc, kBindBufferInt64,
        FunctionType::get(
            {ArrayRef<Type>{llvmPointerType, llvmInt32Type, llvmInt32Type,
                            mlirUnrankedMemRefI64Type}},
            {}, &ctx),
        ArrayRef<std::pair<mlir::Identifier, mlir::Attribute>>());
  }

  if (!module.lookupSymbol(kInitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kInitVulkan,
        LLVM::LLVMType::getFunctionTy(llvmPointerType, {},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kDeinitVulkan)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kDeinitVulkan,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kCreateLaunchKernelAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kCreateLaunchKernelAction,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kAddVulkanLaunchActionToSchedule)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kAddVulkanLaunchActionToSchedule,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSubmitCommandBuffers)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSubmitCommandBuffers,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kCreateMemoryTransferAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kCreateMemoryTransferAction,
        LLVM::LLVMType::getFunctionTy(llvmVoidType,
                                      {llvmPointerType, llvmInt64Type,
                                       llvmInt64Type, llvmInt64Type,
                                       llvmInt64Type},
                                      /*isVarArg=*/false));
  }

  if (!module.lookupSymbol(kSetLaunchKernelAction)) {
    builder.create<LLVM::LLVMFuncOp>(
        loc, kSetLaunchKernelAction,
        LLVM::LLVMType::getFunctionTy(llvmVoidType, {llvmPointerType},
                                      /*isVarArg=*/false));
  }
}

void ConvertGpuLaunchFuncToVulkanLaunchFunc::convertGpuLaunchFunc(
    gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Create call to `initVulkan` before the first GpuLauchFunc.
  if (lauchFuncIndex == 0) {
    auto initVulkanCall = builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{llvmPointerType},
        builder.getSymbolRefAttr(kInitVulkan), ArrayRef<Value>{});
    vulkanRuntime = initVulkanCall.getResult(0);
  }

  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{llvmVoidType},
      builder.getSymbolRefAttr(kCreateLaunchKernelAction),
      ArrayRef<Value>{vulkanRuntime});

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(createBinaryShader(module, binary)))
    return signalPassFailure();

  // Declare vulkan launch function.
  if (failed(declareVulkanLaunchFunc(loc, launchOp)))
    return signalPassFailure();

  auto operands = SmallVector<Value, 4>(launchOp.getOperands());

  operands.insert(operands.begin(), vulkanRuntime);
  // Create vulkan launch call op.
  auto vulkanLaunchCallOp = builder.create<CallOp>(
      loc, ArrayRef<Type>{},
      builder.getSymbolRefAttr(kVulkanLaunch + std::to_string(lauchFuncIndex)),
      operands);

  // Set SPIR-V binary shader data as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVBlobAttrName,
      StringAttr::get({binary.data(), binary.size()}, loc->getContext()));

  // Set entry point name as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVEntryPointAttrName,
      StringAttr::get(launchOp.kernel(), loc->getContext()));

  auto buffers = launchOp.operands();

  // Create LLVM constant for the descriptor set index.
  // Bind all memrefs to the `0` descriptor set, the same way as `GPUToSPIRV`
  // pass does.
  Value descriptorSet = builder.create<LLVM::ConstantOp>(
      loc, llvmInt32Type, builder.getI32IntegerAttr(0));

  uint32_t bindIndex = 0;
  for (auto buffer : buffers) {
    // Create LLVM constant for the descriptor binding index.
    Value descriptorBinding = builder.create<LLVM::ConstantOp>(
        loc, llvmInt32Type, builder.getI32IntegerAttr(bindIndex));

    auto type = buffer.getType();
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>()) {
      auto rank = memRefType.getRank();
      auto elemntType = memRefType.getElementType();
      if (elemntType.isInteger(64)) {
        mlir::Type memRefType;
        if (rank == 1)
          memRefType = mlir1DDynamicMemRefI64Type;
        if (rank == 2)
          memRefType = mlir2DDynamicMemRefI64Type;
        if (rank == 3)
          memRefType = mlir3DDynamicMemRefI64Type;
        auto dynamicBuffer =
            builder.create<mlir::MemRefCastOp>(loc, buffer, memRefType);
        Value unrankedBuffer = builder.create<mlir::MemRefCastOp>(
            loc, dynamicBuffer, mlirUnrankedMemRefI64Type);

        builder.create<CallOp>(
            loc, ArrayRef<Type>{}, builder.getSymbolRefAttr(kBindBufferInt64),
            ArrayRef<Value>{vulkanRuntime, descriptorSet, descriptorBinding,
                            unrankedBuffer});
      }
      if (elemntType.isF32()) {
        mlir::Type memRefType;
        if (rank == 1)
          memRefType = mlir1DDynamicMemRefF32Type;
        if (rank == 2)
          memRefType = mlir2DDynamicMemRefF32Type;
        if (rank == 3)
          memRefType = mlir3DDynamicMemRefF32Type;

        auto dynamicBuffer =
            builder.create<mlir::MemRefCastOp>(loc, buffer, memRefType);
        Value unrankedBuffer = builder.create<mlir::MemRefCastOp>(
            loc, dynamicBuffer, mlirUnrankedMemRefF32Type);

        builder.create<CallOp>(
            loc, ArrayRef<Type>{}, builder.getSymbolRefAttr(kBindBufferFloat32),
            ArrayRef<Value>{vulkanRuntime, descriptorSet, descriptorBinding,
                            unrankedBuffer});
      }
    }

    bindIndex++;
  }

  // Create call to `setLaunchKernelAction` runtime function.
  builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                               builder.getSymbolRefAttr(kSetLaunchKernelAction),
                               ArrayRef<Value>{vulkanRuntime});

  for (size_t i = 0; i < buffers.size(); i++) {
    for (auto pair : bufferMap) {
      if (pair.first == buffers[i]) {
        Value dst_index = builder.create<LLVM::ConstantOp>(
            loc, llvmInt64Type, builder.getI64IntegerAttr(lauchFuncIndex));

        Value dst_binding = builder.create<LLVM::ConstantOp>(
            loc, llvmInt64Type, builder.getI64IntegerAttr(i));

        Value src_index = builder.create<LLVM::ConstantOp>(
            loc, llvmInt64Type, builder.getI64IntegerAttr(pair.second[0]));

        Value src_binding = builder.create<LLVM::ConstantOp>(
            loc, llvmInt64Type, builder.getI64IntegerAttr(pair.second[1]));

        builder.create<LLVM::CallOp>(
            loc, ArrayRef<Type>{llvmVoidType},
            builder.getSymbolRefAttr(kCreateMemoryTransferAction),
            ArrayRef<Value>{vulkanRuntime, src_index, src_binding, dst_index,
                            dst_binding});
      }
    }
    llvm::SmallVector<uint, 2> second;
    second.push_back(lauchFuncIndex);
    second.push_back(i);
    bufferMap[buffers[i]] = second;
  }

  builder.create<LLVM::CallOp>(
      loc, ArrayRef<Type>{llvmVoidType},
      builder.getSymbolRefAttr(kAddVulkanLaunchActionToSchedule),
      ArrayRef<Value>{vulkanRuntime});

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  if (lauchFuncIndex == numKernel - 1) {
    // Create call to 'submitCommandBuffers' runtime function.
    builder.create<LLVM::CallOp>(
        loc, ArrayRef<Type>{llvmVoidType},
        builder.getSymbolRefAttr(kSubmitCommandBuffers),
        ArrayRef<Value>{vulkanRuntime});

    // Create call to 'deinitVulkan' runtime function.
    builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{llvmVoidType},
                                 builder.getSymbolRefAttr(kDeinitVulkan),
                                 ArrayRef<Value>{vulkanRuntime});
  }

  if (VLOG_IS_ON(4)) {
    if (failed(printLauchOpBuffers(loc, builder, launchOp))) {
      return signalPassFailure();
    }
  }

  launchOp.erase();
  lauchFuncIndex++;
}

namespace pmlc::conversion::gpu {

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanCallsPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}

} // namespace pmlc::conversion::gpu

static PassRegistration<ConvertGpuLaunchFuncToVulkanLaunchFunc>
    pass("pmlc-convert-gpu-to-vulkan",
         "Convert gpu.launch_func op to Vulkan runtime calls");
