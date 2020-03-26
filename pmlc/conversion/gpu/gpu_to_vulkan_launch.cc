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
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

static constexpr const char *kSPIRVBlobAttrName = "spirv_blob";
static constexpr const char *kSPIRVEntryPointAttrName = "spirv_entry_point";
static constexpr const char *kVulkanLaunch = "vulkanLaunch";
static constexpr const char *kPrint_memref_f32 = "print_memref_f32";

namespace {

static int gpu_launch_index = 0;

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
    // TODO(denis0x0D): Handle other types.
    if (auto memRefType = type.dyn_cast_or_null<MemRefType>())
      return memRefType.hasRank() && memRefType.getRank() == 1;
    return false;
  }

  void getMlirTypes() {
    OpBuilder builder(getModule());
    mlirInt32Type = builder.getIntegerType(32);
    mlirFloat32Type = builder.getF32Type();
    mlirUnrankedMemRefF32Type =
        UnrankedMemRefType::get(mlirFloat32Type, /*memorySpace=*/0);

    SmallVector<int64_t, 4> shapeConstants1D = {-1};
    mlir1DDynamicMemRefF32Type =
        MemRefType::get(shapeConstants1D, mlirFloat32Type);

    SmallVector<int64_t, 4> shapeConstants2D = {-1, -1};
    mlir2DDynamicMemRefF32Type =
        MemRefType::get(shapeConstants2D, mlirFloat32Type);

    SmallVector<int64_t, 4> shapeConstants3D = {-1, -1, -1};
    mlir3DDynamicMemRefF32Type =
        MemRefType::get(shapeConstants3D, mlirFloat32Type);

    SmallVector<int64_t, 4> shapeConstants4D = {-1, -1, -1, -1};
    mlir4DDynamicMemRefF32Type =
        MemRefType::get(shapeConstants4D, mlirFloat32Type);
  }
  /// Declares the vulkan launch function. Returns an error if the any type of
  /// operand is unsupported by Vulkan runtime.
  LogicalResult declareVulkanLaunchFunc(Location loc,
                                        gpu::LaunchFuncOp launchOp);

  mlir::Type mlirInt32Type;
  mlir::Type mlirFloat32Type;
  mlir::Type mlir1DDynamicMemRefF32Type;
  mlir::Type mlir2DDynamicMemRefF32Type;
  mlir::Type mlir3DDynamicMemRefF32Type;
  mlir::Type mlir4DDynamicMemRefF32Type;
  mlir::Type mlirUnrankedMemRefF32Type;
};

} // anonymous namespace

void ConvertGpuLaunchFuncToVulkanLaunchFunc::runOnModule() {
  getMlirTypes();

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

  // Check that all operands have supported types except those for the launch
  // configuration.
  /*
  for (auto type : llvm::drop_begin(vulkanLaunchTypes, 6)) {
    if (!isSupportedType(type))
      return launchOp.emitError() << type << " is unsupported to run on Vulkan";
  }
  */
  // Declare vulkan launch function.
  builder.create<FuncOp>(
      loc, kVulkanLaunch + std::to_string(gpu_launch_index),
      FunctionType::get(vulkanLaunchTypes, ArrayRef<Type>{}, loc->getContext()),
      ArrayRef<NamedAttribute>{});

  return success();
}

LogicalResult ConvertGpuLaunchFuncToVulkanLaunchFunc::createBinaryShader(
    ModuleOp module, std::vector<char> &binaryShader) {

  SmallVector<uint32_t, 0> binary;
  auto shader_index = 0;
  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (shader_index == gpu_launch_index) {
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
    auto dynamicBuffer = builder.create<mlir::MemRefCastOp>(
        loc, buffer, mlir2DDynamicMemRefF32Type);
    auto unrankedBuffer = builder.create<mlir::MemRefCastOp>(
        loc, dynamicBuffer, mlirUnrankedMemRefF32Type);

    builder.create<CallOp>(loc, ArrayRef<Type>{},
                           builder.getSymbolRefAttr(kPrint_memref_f32),
                           ArrayRef<Value>(unrankedBuffer));
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
}

void ConvertGpuLaunchFuncToVulkanLaunchFunc::convertGpuLaunchFunc(
    gpu::LaunchFuncOp launchOp) {
  ModuleOp module = getModule();
  OpBuilder builder(launchOp);
  Location loc = launchOp.getLoc();

  // Serialize `spirv::Module` into binary form.
  std::vector<char> binary;
  if (failed(createBinaryShader(module, binary)))
    return signalPassFailure();

  // Declare vulkan launch function.
  if (failed(declareVulkanLaunchFunc(loc, launchOp)))
    return signalPassFailure();

  // Create vulkan launch call op.
  auto vulkanLaunchCallOp = builder.create<CallOp>(
      loc, ArrayRef<Type>{},
      builder.getSymbolRefAttr(kVulkanLaunch +
                               std::to_string(gpu_launch_index)),
      launchOp.getOperands());

  // Set SPIR-V binary shader data as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVBlobAttrName,
      StringAttr::get({binary.data(), binary.size()}, loc->getContext()));

  // Set entry point name as an attribute.
  vulkanLaunchCallOp.setAttr(
      kSPIRVEntryPointAttrName,
      StringAttr::get(launchOp.kernel(), loc->getContext()));

  // Print buffers after vulkan launch call.

  if (VLOG_IS_ON(1)) {
    if (failed(printLauchOpBuffers(loc, builder, launchOp))) {
      return signalPassFailure();
    }
  }

  // Declare runtime functions.
  declareVulkanFunctions(loc);

  launchOp.erase();

  gpu_launch_index++;
}

namespace pmlc::conversion::gpu {

std::unique_ptr<mlir::Pass> createConvertGpuLaunchFuncToVulkanCallsPass() {
  return std::make_unique<ConvertGpuLaunchFuncToVulkanLaunchFunc>();
}

} // namespace pmlc::conversion::gpu

static PassRegistration<ConvertGpuLaunchFuncToVulkanLaunchFunc>
    pass("pmlc-convert-gpu-to-vulkan",
         "Convert gpu.launch_func op to Vulkan runtime calls");
